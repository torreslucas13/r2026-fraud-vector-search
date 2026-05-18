/*
 * build_index: parse references.json.gz, quantize 14-dim vectors to int16,
 * cluster with k-means (IVF), dump a binary index for the API to mmap.
 *
 * Output layout (little-endian) — magic "RINHA005":
 *   [8]              magic "RINHA005"
 *   [4]   uint32     count_real       (3,000,000)
 *   [4]   uint32     n_clusters       (e.g. 2048; MUST be multiple of 8)
 *   [4]   uint32     n_groups         (sum of ceil(n_real_c / 8) over clusters)
 *   [4]              padding
 *   [n_clusters/8 * 256] centroid_groups (i16, pair-packed SoA — same kernel as vec groups)
 *   [n_clusters*28]  bbox_min         (int16, 14 dims per cluster — min[d] over members)
 *   [n_clusters*28]  bbox_max         (int16, 14 dims per cluster — max[d] over members)
 *   [(n_clusters+1)*4] group_offsets  (uint32; cumulative group counts)
 *   [(n_clusters+1)*4] real_offsets   (uint32; cumulative real-vector counts; for last-group mask)
 *   [n_groups*256]   group_blocks     (i16, pair-packed SoA: per group, 8 dim-pairs ×
 *                                       (8 vecs × (d_p0, d_p1)) = 128 i16 = 256 bytes)
 *   [n_groups*8]     labels           (uint8 per slot; padded slots = 0, masked at scan time)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <zlib.h>

#define VEC_DIMS 14
#define STRIDE 16
#define SCALE 10000

#define N_CLUSTERS 4096
#define SAMPLE_SIZE 100000
#define KMEANS_ITER 6

#define BUFSZ (1 << 20)
typedef struct {
    gzFile gz;
    char buf[BUFSZ + 1];
    int len;
    int pos;
    int eof;
} Reader;

static int reader_refill(Reader *r) {
    if (r->pos < r->len) {
        memmove(r->buf, r->buf + r->pos, r->len - r->pos);
        r->len -= r->pos;
        r->pos = 0;
    } else {
        r->len = 0;
        r->pos = 0;
    }
    if (r->eof) return r->len > 0;
    int n = gzread(r->gz, r->buf + r->len, BUFSZ - r->len);
    if (n <= 0) { r->eof = 1; return r->len > 0; }
    r->len += n;
    return 1;
}
static int reader_peek(Reader *r, char *out) {
    if (r->pos >= r->len && !reader_refill(r)) return 0;
    *out = r->buf[r->pos];
    return 1;
}
static void reader_advance(Reader *r) {
    if (r->pos < r->len) r->pos++;
    else if (reader_refill(r) && r->pos < r->len) r->pos++;
}
static char skip_until_any(Reader *r, const char *set) {
    char c;
    while (reader_peek(r, &c)) {
        if (strchr(set, c)) return c;
        reader_advance(r);
    }
    return 0;
}
static int parse_double(Reader *r, double *out) {
    if (r->len - r->pos < 64 && !r->eof) reader_refill(r);
    r->buf[r->len] = '\0';
    char *start = r->buf + r->pos;
    char *end;
    *out = strtod(start, &end);
    if (end == start) return 0;
    r->pos += (int)(end - start);
    return 1;
}
static int16_t quantize(double x) {
    if (x < -1.0) x = -1.0;
    if (x > 1.0) x = 1.0;
    long v = (long)(x * SCALE + (x >= 0 ? 0.5 : -0.5));
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

/* ---------- k-means ---------- */

static uint32_t xorshift_state = 0x12345678u;
static uint32_t xrand(void) {
    uint32_t x = xorshift_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    xorshift_state = x;
    return x;
}

static int32_t sq_dist(const int16_t *a, const int16_t *b) {
    int32_t d = 0;
    for (int j = 0; j < VEC_DIMS; j++) {
        int32_t diff = (int32_t)a[j] - (int32_t)b[j];
        d += diff * diff;
    }
    return d;
}

static int32_t nearest_centroid(const int16_t *q, const int16_t *centroids, int K) {
    int32_t best_d = INT32_MAX;
    int best_i = 0;
    for (int c = 0; c < K; c++) {
        int32_t d = sq_dist(q, centroids + (size_t)c * STRIDE);
        if (d < best_d) { best_d = d; best_i = c; }
    }
    return best_i;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <references.json.gz> <index.bin>\n", argv[0]);
        return 1;
    }
    Reader r = {0};
    r.gz = gzopen(argv[1], "rb");
    if (!r.gz) { perror("gzopen"); return 1; }

    /* ----- Pass 1: parse all vectors into memory ----- */
    fprintf(stderr, "parsing references...\n");
    size_t cap = 1 << 20;
    int16_t *vectors = malloc(cap * STRIDE * sizeof(int16_t));
    uint8_t *labels  = malloc(cap);
    if (!vectors || !labels) { perror("malloc"); return 1; }
    uint32_t count = 0;

    char c;
    if (skip_until_any(&r, "[") != '[') { fprintf(stderr, "no [\n"); return 1; }
    reader_advance(&r);

    while (1) {
        c = skip_until_any(&r, "{]");
        if (c == ']' || c == 0) break;
        reader_advance(&r);

        c = skip_until_any(&r, "[");
        if (c != '[') { fprintf(stderr, "missing vector\n"); return 1; }
        reader_advance(&r);

        if (count >= cap) {
            cap *= 2;
            vectors = realloc(vectors, cap * STRIDE * sizeof(int16_t));
            labels  = realloc(labels,  cap);
            if (!vectors || !labels) { perror("realloc"); return 1; }
        }
        int16_t *vec = vectors + (size_t)count * STRIDE;
        memset(vec, 0, STRIDE * sizeof(int16_t));

        for (int d = 0; d < VEC_DIMS; d++) {
            while (reader_peek(&r, &c) && (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == ',')) {
                reader_advance(&r);
            }
            double v;
            if (!parse_double(&r, &v)) {
                fprintf(stderr, "parse fail at dim %d, count=%u\n", d, count);
                return 1;
            }
            vec[d] = quantize(v);
        }

        c = skip_until_any(&r, "]"); if (c != ']') return 1;
        reader_advance(&r);
        c = skip_until_any(&r, ":"); if (c != ':') return 1;
        reader_advance(&r);
        c = skip_until_any(&r, "\""); if (c != '"') return 1;
        reader_advance(&r);
        if (!reader_peek(&r, &c)) return 1;
        labels[count] = (c == 'f') ? 1 : 0;
        while (reader_peek(&r, &c) && c != '"') reader_advance(&r);
        reader_advance(&r);
        c = skip_until_any(&r, "}"); if (c != '}') return 1;
        reader_advance(&r);

        count++;
        if ((count % 500000) == 0) fprintf(stderr, "  parsed %u vectors\n", count);
    }
    gzclose(r.gz);
    fprintf(stderr, "parsed %u vectors total\n", count);

    /* ----- k-means: train centroids on a sample ----- */
    const int K = N_CLUSTERS;
    int16_t *centroids = calloc((size_t)K * STRIDE, sizeof(int16_t));
    if (!centroids) { perror("calloc centroids"); return 1; }

    /* Pick K random distinct samples as initial centroids. */
    fprintf(stderr, "initializing %d centroids from random samples...\n", K);
    xorshift_state = 0xdeadbeefu;
    for (int c2 = 0; c2 < K; c2++) {
        uint32_t idx = xrand() % count;
        memcpy(centroids + (size_t)c2 * STRIDE,
               vectors   + (size_t)idx * STRIDE,
               STRIDE * sizeof(int16_t));
    }

    /* Sample indices for training. */
    int sample_n = SAMPLE_SIZE < (int)count ? SAMPLE_SIZE : (int)count;
    uint32_t *sample = malloc((size_t)sample_n * sizeof(uint32_t));
    if (!sample) { perror("malloc sample"); return 1; }
    for (int i = 0; i < sample_n; i++) sample[i] = xrand() % count;

    int64_t *sum = malloc((size_t)K * VEC_DIMS * sizeof(int64_t));
    int32_t *cnt = malloc((size_t)K * sizeof(int32_t));
    if (!sum || !cnt) { perror("malloc kmeans"); return 1; }

    for (int it = 0; it < KMEANS_ITER; it++) {
        memset(sum, 0, (size_t)K * VEC_DIMS * sizeof(int64_t));
        memset(cnt, 0, (size_t)K * sizeof(int32_t));
        for (int i = 0; i < sample_n; i++) {
            const int16_t *v = vectors + (size_t)sample[i] * STRIDE;
            int c2 = nearest_centroid(v, centroids, K);
            cnt[c2]++;
            int64_t *s = sum + (size_t)c2 * VEC_DIMS;
            for (int j = 0; j < VEC_DIMS; j++) s[j] += v[j];
        }
        for (int c2 = 0; c2 < K; c2++) {
            int16_t *cn = centroids + (size_t)c2 * STRIDE;
            if (cnt[c2] == 0) {
                /* dead centroid — reseed from a random sample */
                uint32_t idx = xrand() % count;
                memcpy(cn, vectors + (size_t)idx * STRIDE, STRIDE * sizeof(int16_t));
                continue;
            }
            int64_t *s = sum + (size_t)c2 * VEC_DIMS;
            for (int j = 0; j < VEC_DIMS; j++) cn[j] = (int16_t)(s[j] / cnt[c2]);
        }
        fprintf(stderr, "  k-means iter %d/%d\n", it + 1, KMEANS_ITER);
    }
    free(sample); free(sum); free(cnt);

    /* ----- Assign all vectors to nearest centroid ----- */
    fprintf(stderr, "assigning %u vectors to %d clusters...\n", count, K);
    uint32_t *assign = malloc((size_t)count * sizeof(uint32_t));
    if (!assign) { perror("malloc assign"); return 1; }
    int32_t *cluster_count = calloc((size_t)K, sizeof(int32_t));
    if (!cluster_count) { perror("calloc cluster_count"); return 1; }
    for (uint32_t i = 0; i < count; i++) {
        const int16_t *v = vectors + (size_t)i * STRIDE;
        int c2 = nearest_centroid(v, centroids, K);
        assign[i] = (uint32_t)c2;
        cluster_count[c2]++;
        if ((i % 500000) == 499999) fprintf(stderr, "  assigned %u/%u\n", i + 1, count);
    }

    /* Compute real- and group-offsets (cumulative). */
    uint32_t *real_offsets  = malloc((size_t)(K + 1) * sizeof(uint32_t));
    uint32_t *group_offsets = malloc((size_t)(K + 1) * sizeof(uint32_t));
    if (!real_offsets || !group_offsets) { perror("malloc offsets"); return 1; }
    real_offsets[0] = 0;
    group_offsets[0] = 0;
    for (int c2 = 0; c2 < K; c2++) {
        real_offsets[c2 + 1]  = real_offsets[c2]  + (uint32_t)cluster_count[c2];
        uint32_t ng = ((uint32_t)cluster_count[c2] + 7) / 8;
        group_offsets[c2 + 1] = group_offsets[c2] + ng;
    }
    uint32_t n_groups = group_offsets[K];

    /* Place vectors into per-cluster bucket arrays (AoS, sorted by cluster). */
    int16_t *sorted_vec = malloc((size_t)count * STRIDE * sizeof(int16_t));
    uint8_t *sorted_lbl = malloc((size_t)count);
    if (!sorted_vec || !sorted_lbl) { perror("malloc sorted"); return 1; }
    uint32_t *cur = calloc((size_t)K, sizeof(uint32_t));
    if (!cur) { perror("calloc cur"); return 1; }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t c2 = assign[i];
        uint32_t dst = real_offsets[c2] + cur[c2]++;
        memcpy(sorted_vec + (size_t)dst * STRIDE,
               vectors    + (size_t)i   * STRIDE,
               STRIDE * sizeof(int16_t));
        sorted_lbl[dst] = labels[i];
    }
    free(cur); free(assign);
    free(vectors); free(labels);

    /* ----- Bbox per cluster ----- */
    fprintf(stderr, "computing bboxes...\n");
    int16_t *bbox_min = malloc((size_t)K * VEC_DIMS * sizeof(int16_t));
    int16_t *bbox_max = malloc((size_t)K * VEC_DIMS * sizeof(int16_t));
    if (!bbox_min || !bbox_max) { perror("malloc bbox"); return 1; }
    for (int c2 = 0; c2 < K; c2++) {
        int16_t *mn = bbox_min + (size_t)c2 * VEC_DIMS;
        int16_t *mx = bbox_max + (size_t)c2 * VEC_DIMS;
        for (int d = 0; d < VEC_DIMS; d++) { mn[d] = INT16_MAX; mx[d] = INT16_MIN; }
        uint32_t s = real_offsets[c2], e = real_offsets[c2 + 1];
        if (s == e) {
            /* Empty cluster: keep MAX/MIN so lower-bound becomes huge and prunes. */
            continue;
        }
        for (uint32_t i = s; i < e; i++) {
            const int16_t *v = sorted_vec + (size_t)i * STRIDE;
            for (int d = 0; d < VEC_DIMS; d++) {
                if (v[d] < mn[d]) mn[d] = v[d];
                if (v[d] > mx[d]) mx[d] = v[d];
            }
        }
    }

    /* ----- Pack vectors into pair-packed SoA groups -----
     * Per group (8 vectors v[0..7], 16 dims each):
     *   for pair p in 0..7 (dims 2p, 2p+1):
     *     16 i16 = (v[0][2p], v[0][2p+1], v[1][2p], v[1][2p+1], ..., v[7][2p], v[7][2p+1])
     *
     * Padded slots (k >= n_real_in_group of the last group of each cluster) are
     * filled with the first real vector of the same group: kernel computes a
     * valid (possibly small) distance, scan loop ignores those slots.
     */
    fprintf(stderr, "packing %u groups...\n", n_groups);
    int16_t *groups = calloc((size_t)n_groups * 128, sizeof(int16_t)); /* 128 i16 / group */
    uint8_t *group_labels = calloc((size_t)n_groups * 8, 1);
    if (!groups || !group_labels) { perror("calloc groups"); return 1; }

    for (int c2 = 0; c2 < K; c2++) {
        uint32_t n_real = real_offsets[c2 + 1] - real_offsets[c2];
        if (n_real == 0) continue;
        uint32_t start = real_offsets[c2];
        uint32_t gs    = group_offsets[c2];
        uint32_t ng    = group_offsets[c2 + 1] - gs;

        for (uint32_t g = 0; g < ng; g++) {
            int16_t *gp = groups + (size_t)(gs + g) * 128;
            uint8_t *lp = group_labels + (size_t)(gs + g) * 8;
            uint32_t base = start + g * 8;
            uint32_t valid = (g + 1 == ng) ? (n_real - g * 8) : 8;
            /* For each pair p (dims 2p, 2p+1), interleave across 8 vector slots. */
            for (int p = 0; p < 8; p++) {
                int d0 = 2 * p, d1 = 2 * p + 1;
                for (int k = 0; k < 8; k++) {
                    uint32_t src = (k < (int)valid) ? (base + k) : base; /* dup slot 0 for pad */
                    const int16_t *v = sorted_vec + (size_t)src * STRIDE;
                    gp[p * 16 + k * 2 + 0] = v[d0];
                    gp[p * 16 + k * 2 + 1] = v[d1];
                }
            }
            for (uint32_t k = 0; k < valid; k++) {
                lp[k] = sorted_lbl[base + k];
            }
            /* Padded labels stay 0; scan loop bounds at `valid`. */
        }
    }

    free(sorted_vec); free(sorted_lbl); free(cluster_count);

    /* ----- Pack centroids into pair-packed SoA groups (same layout as vec groups). */
    if ((K & 7) != 0) {
        fprintf(stderr, "N_CLUSTERS must be a multiple of 8\n");
        return 1;
    }
    int n_cgrp = K / 8;
    int16_t *centroid_groups = calloc((size_t)n_cgrp * 128, sizeof(int16_t));
    if (!centroid_groups) { perror("calloc centroid_groups"); return 1; }
    for (int g = 0; g < n_cgrp; g++) {
        int16_t *gp = centroid_groups + (size_t)g * 128;
        for (int p = 0; p < 8; p++) {
            int d0 = 2 * p, d1 = 2 * p + 1;
            for (int k = 0; k < 8; k++) {
                const int16_t *v = centroids + (size_t)(g * 8 + k) * STRIDE;
                gp[p * 16 + k * 2 + 0] = v[d0];
                gp[p * 16 + k * 2 + 1] = v[d1];
            }
        }
    }

    /* ----- Write output ----- */
    FILE *out = fopen(argv[2], "wb");
    if (!out) { perror("fopen out"); return 1; }
    fwrite("RINHA005", 1, 8, out);
    fwrite(&count, 4, 1, out);
    uint32_t k32 = (uint32_t)K;
    fwrite(&k32, 4, 1, out);
    fwrite(&n_groups, 4, 1, out);
    uint32_t pad32 = 0;
    fwrite(&pad32, 4, 1, out);
    fwrite(centroid_groups, sizeof(int16_t), (size_t)n_cgrp * 128, out);
    fwrite(bbox_min,      sizeof(int16_t), (size_t)K * VEC_DIMS, out);
    fwrite(bbox_max,      sizeof(int16_t), (size_t)K * VEC_DIMS, out);
    fwrite(group_offsets, sizeof(uint32_t), (size_t)(K + 1), out);
    fwrite(real_offsets,  sizeof(uint32_t), (size_t)(K + 1), out);
    fwrite(groups,        sizeof(int16_t), (size_t)n_groups * 128, out);
    fwrite(group_labels,  1, (size_t)n_groups * 8, out);
    fclose(out);

    free(centroids); free(centroid_groups);
    free(bbox_min); free(bbox_max);
    free(group_offsets); free(real_offsets);
    free(groups); free(group_labels);
    fprintf(stderr, "done: %u vectors in %d clusters (%u groups)\n", count, K, n_groups);
    return 0;
}
