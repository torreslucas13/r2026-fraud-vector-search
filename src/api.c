/*
 * api: epoll-based HTTP/1.1 server for POST /fraud-score and GET /ready.
 *
 * Memory-maps index.bin (built by build_index), brute-force scans all N
 * vectors per query, tracks top-5 nearest by squared L2, counts fraud labels.
 *
 * MVP: scalar brute force. SIMD/IVF come later.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/un.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#define VEC_DIMS 14
#define STRIDE 16   /* each vector occupies 16 i16 lanes for AVX2 alignment */
#define SCALE 10000
#define K 5
#define MAX_CLUSTERS 4096   /* upper bound for stack arrays — actual G_nclusters = 2048 */
/* Two-phase: fast IVF top-NPROBE_FAST, then a wider sweep on decision-boundary
 * fraud counts {2,3}. Per-cluster bbox lower-bound prune skips clusters whose
 * closest possible point is already worse than the current 5th-best. */
#define NPROBE_FAST   24
#define NPROBE_REPAIR 192
#define MAX_BODY 4096
#define MAX_CONNS 4096
#define EVBATCH 256

/* Normalization constants (from resources/normalization.json). */
#define MAX_AMOUNT 10000.0
#define MAX_INSTALLMENTS 12.0
#define AMOUNT_VS_AVG_RATIO 10.0
#define MAX_MINUTES 1440.0
#define MAX_KM 1000.0
#define MAX_TX_COUNT_24H 20.0
#define MAX_MERCHANT_AVG_AMOUNT 10000.0

/* MCC risk table (from resources/mcc_risk.json). Linear scan: only 10 entries. */
static const struct { const char *mcc; double risk; } MCC_TABLE[] = {
    {"5411", 0.15}, {"5812", 0.30}, {"5912", 0.20}, {"5944", 0.45},
    {"7801", 0.80}, {"7802", 0.75}, {"7995", 0.85}, {"4511", 0.35},
    {"5311", 0.25}, {"5999", 0.50},
};
#define MCC_TABLE_SIZE (sizeof(MCC_TABLE) / sizeof(MCC_TABLE[0]))
#define MCC_DEFAULT 0.5

/* Pre-rendered HTTP responses for fraud_count 0..5. */
static const char *RESP[6] = {
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
};
static int RESP_LEN[6];

static const char *READY_RESP =
    "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: text/plain\r\n\r\nok";
static int READY_RESP_LEN;

static const char *BAD_RESP =
    "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n";
static int BAD_RESP_LEN;

/* Index data, mmap'd (format RINHA005). */
static const int16_t  *G_centroid_groups; /* 128 i16 per group, n_clusters/8 groups */
static const int16_t  *G_bbox_min;
static const int16_t  *G_bbox_max;
static const uint32_t *G_group_offsets;  /* in groups */
static const uint32_t *G_real_offsets;   /* in real vectors */
static const int16_t  *G_groups;         /* 128 i16 per group, pair-packed SoA */
static const uint8_t  *G_labels;         /* one byte per slot (n_groups * 8 entries) */
static uint32_t G_count;
static uint32_t G_nclusters;
static uint32_t G_ngroups;

/* ---------- JSON helpers ---------- */

/* Find `"key":` inside [buf, buf+len). Return pointer just past the colon
 * (skipping spaces), or NULL. */
static const char *find_key(const char *buf, int len, const char *key) {
    /* needle = "key" (with quotes). Then we tolerate space before colon. */
    char needle[64];
    int nlen = snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *end = buf + len;
    const char *p = buf;
    while (p < end - nlen) {
        const char *m = memmem(p, end - p, needle, nlen);
        if (!m) return NULL;
        const char *q = m + nlen;
        while (q < end && (*q == ' ' || *q == '\t')) q++;
        if (q < end && *q == ':') {
            q++;
            while (q < end && (*q == ' ' || *q == '\t')) q++;
            return q;
        }
        p = m + 1;
    }
    return NULL;
}

static double parse_num(const char *p, const char *end) {
    char tmp[64];
    int n = end - p;
    if (n > 63) n = 63;
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    return strtod(tmp, NULL);
}

static double get_num(const char *body, int blen, const char *key, double dflt) {
    const char *p = find_key(body, blen, key);
    if (!p) return dflt;
    return parse_num(p, body + blen);
}

static int get_bool(const char *body, int blen, const char *key, int dflt) {
    const char *p = find_key(body, blen, key);
    if (!p) return dflt;
    return (*p == 't') ? 1 : 0;
}

/* Returns pointer to opening quote of string value, or NULL. */
static const char *get_str(const char *body, int blen, const char *key, int *out_len) {
    const char *p = find_key(body, blen, key);
    if (!p || *p != '"') return NULL;
    const char *s = p + 1;
    const char *e = memchr(s, '"', body + blen - s);
    if (!e) return NULL;
    *out_len = (int)(e - s);
    return s;
}

static int is_null(const char *body, int blen, const char *key) {
    const char *p = find_key(body, blen, key);
    return p && *p == 'n';
}

/* ---------- Vectorize ---------- */

static double clamp01(double x) {
    if (x < 0) return 0;
    if (x > 1) return 1;
    return x;
}

static int16_t quant(double x) {
    long v = (long)(x * SCALE + (x >= 0 ? 0.5 : -0.5));
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    return (int16_t)v;
}

/* Tomohiko Sakamoto day-of-week. Monday=0 .. Sunday=6. */
static int day_of_week(int y, int m, int d) {
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    y -= (m < 3);
    int n = (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7;
    /* Algorithm yields Sun=0..Sat=6 — shift to Mon=0..Sun=6. */
    return (n + 6) % 7;
}

/* Parse ISO-8601 "YYYY-MM-DDTHH:MM:SSZ" → fills *hour and *dow. */
static void parse_timestamp(const char *s, int *hour, int *dow) {
    int y = (s[0]-'0')*1000 + (s[1]-'0')*100 + (s[2]-'0')*10 + (s[3]-'0');
    int m = (s[5]-'0')*10 + (s[6]-'0');
    int d = (s[8]-'0')*10 + (s[9]-'0');
    *hour = (s[11]-'0')*10 + (s[12]-'0');
    *dow = day_of_week(y, m, d);
}

static double mcc_risk(const char *mcc, int mcclen) {
    for (size_t i = 0; i < MCC_TABLE_SIZE; i++) {
        if (mcclen == 4 && memcmp(mcc, MCC_TABLE[i].mcc, 4) == 0) {
            return MCC_TABLE[i].risk;
        }
    }
    return MCC_DEFAULT;
}

/* Vectorize body into 16 int16 lanes (14 real + 2 zero pad). */
static void vectorize(const char *body, int blen, int16_t out[STRIDE]) {
    out[14] = 0;
    out[15] = 0;
    /* transaction.* */
    double amount = get_num(body, blen, "amount", 0);
    double installments = get_num(body, blen, "installments", 1);
    /* requested_at */
    int rlen = 0;
    const char *rs = get_str(body, blen, "requested_at", &rlen);
    int hour = 0, dow = 0;
    if (rs && rlen >= 19) parse_timestamp(rs, &hour, &dow);

    /* customer.* */
    double avg_amount = get_num(body, blen, "avg_amount", 1);
    double tx_count_24h = get_num(body, blen, "tx_count_24h", 0);

    /* merchant.* — scope to the merchant block. Top-level "id" matches the
     * transaction id instead of merchant.id, so we must not search globally. */
    int mcclen = 0;
    int midlen = 0;
    const char *mcc = NULL;
    const char *mid = NULL;
    double merchant_avg = 0;
    const char *mp = find_key(body, blen, "merchant");
    if (mp) {
        int sublen = body + blen - mp;
        mcc = get_str(mp, sublen, "mcc", &mcclen);
        mid = get_str(mp, sublen, "id", &midlen);
        merchant_avg = get_num(mp, sublen, "avg_amount", 0);
    }

    /* terminal.* */
    int is_online = get_bool(body, blen, "is_online", 0);
    int card_present = get_bool(body, blen, "card_present", 0);
    double km_from_home = get_num(body, blen, "km_from_home", 0);

    /* last_transaction */
    int last_is_null = is_null(body, blen, "last_transaction");
    double km_from_current = 0;
    double minutes_since_last = 0;
    if (!last_is_null) {
        const char *lp = find_key(body, blen, "last_transaction");
        if (lp) {
            int sublen = body + blen - lp;
            km_from_current = get_num(lp, sublen, "km_from_current", 0);
            /* minutes computation: requires diff between requested_at and timestamp.
             * For MVP we skip exact computation and set 0; this is incorrect for
             * the actual challenge — TODO add. */
            int tslen = 0;
            const char *ts = get_str(lp, sublen, "timestamp", &tslen);
            if (ts && rs && tslen >= 19 && rlen >= 19) {
                /* crude minutes diff via tm conversion */
                struct { int Y,M,D,h,m,s; } a, b;
                a.Y=(rs[0]-'0')*1000+(rs[1]-'0')*100+(rs[2]-'0')*10+(rs[3]-'0');
                a.M=(rs[5]-'0')*10+(rs[6]-'0');
                a.D=(rs[8]-'0')*10+(rs[9]-'0');
                a.h=(rs[11]-'0')*10+(rs[12]-'0');
                a.m=(rs[14]-'0')*10+(rs[15]-'0');
                a.s=(rs[17]-'0')*10+(rs[18]-'0');
                b.Y=(ts[0]-'0')*1000+(ts[1]-'0')*100+(ts[2]-'0')*10+(ts[3]-'0');
                b.M=(ts[5]-'0')*10+(ts[6]-'0');
                b.D=(ts[8]-'0')*10+(ts[9]-'0');
                b.h=(ts[11]-'0')*10+(ts[12]-'0');
                b.m=(ts[14]-'0')*10+(ts[15]-'0');
                b.s=(ts[17]-'0')*10+(ts[18]-'0');
                /* days_from_civil for both, then diff in seconds */
                #define DFC(Y,M,D) ({ \
                    int y=(Y)-((M)<=2); int era=(y>=0?y:y-399)/400; \
                    unsigned yoe=(unsigned)(y-era*400); \
                    unsigned moe=(M)+((M)>2?-3:9); \
                    unsigned doe=yoe*365+yoe/4-yoe/100+(153*moe+2)/5+(D)-1; \
                    (long)era*146097 + (long)doe; \
                })
                long da = DFC(a.Y, a.M, a.D);
                long db = DFC(b.Y, b.M, b.D);
                long sa = da*86400 + a.h*3600 + a.m*60 + a.s;
                long sb = db*86400 + b.h*3600 + b.m*60 + b.s;
                long diff = sa - sb;
                if (diff < 0) diff = -diff;
                minutes_since_last = diff / 60.0;
            }
        }
    }

    /* known_merchants check: look for merchant.id substring in known_merchants array. */
    int unknown = 1;
    if (mid && midlen > 0) {
        const char *km = find_key(body, blen, "known_merchants");
        if (km && *km == '[') {
            const char *closebr = memchr(km, ']', body + blen - km);
            if (closebr && memmem(km, closebr - km, mid, midlen)) {
                unknown = 0;
            }
        }
    }

    /* Fill vector. */
    out[0]  = quant(clamp01(amount / MAX_AMOUNT));
    out[1]  = quant(clamp01(installments / MAX_INSTALLMENTS));
    out[2]  = quant(clamp01((amount / (avg_amount > 0 ? avg_amount : 1)) / AMOUNT_VS_AVG_RATIO));
    out[3]  = quant(hour / 23.0);
    out[4]  = quant(dow / 6.0);
    if (last_is_null) {
        out[5] = -SCALE;
        out[6] = -SCALE;
    } else {
        out[5] = quant(clamp01(minutes_since_last / MAX_MINUTES));
        out[6] = quant(clamp01(km_from_current / MAX_KM));
    }
    out[7]  = quant(clamp01(km_from_home / MAX_KM));
    out[8]  = quant(clamp01(tx_count_24h / MAX_TX_COUNT_24H));
    out[9]  = is_online ? SCALE : 0;
    out[10] = card_present ? SCALE : 0;
    out[11] = unknown ? SCALE : 0;
    out[12] = quant(mcc_risk(mcc, mcclen));
    out[13] = quant(clamp01(merchant_avg / MAX_MERCHANT_AVG_AMOUNT));
}

/* ---------- k-NN search (IVF) ----------
 *
 * 1. Compute distance to all centroids, keep top-NPROBE smallest.
 * 2. Scan vectors of those clusters, track top-K nearest neighbors.
 * 3. Return count of fraud labels among them.
 */

/* Lower bound on squared-L2 distance from q to any point in cluster c's bbox.
 * For each dim, the closest in-bbox value to q[d] is clamp(q[d], min[d], max[d]);
 * the contribution is the squared offset of q from that clamped point. Sum 14 dims. */
static inline uint32_t bbox_lb(const int16_t *q, uint32_t c) {
    const int16_t *mn = G_bbox_min + (size_t)c * VEC_DIMS;
    const int16_t *mx = G_bbox_max + (size_t)c * VEC_DIMS;
    uint32_t lb = 0;
    for (int d = 0; d < VEC_DIMS; d++) {
        int32_t qv = q[d];
        int32_t delta = 0;
        if (qv > mx[d]) delta = qv - mx[d];
        else if (qv < mn[d]) delta = mn[d] - qv;
        lb += (uint32_t)(delta * delta);
    }
    return lb;
}

/* Compute squared-L2 distances from q to the 8 vectors in one pair-packed group.
 * Group layout (128 i16): for pair p in 0..7, 16 i16 = interleaved
 *   (v0[d0], v0[d1], v1[d0], v1[d1], ..., v7[d0], v7[d1]) where d0=2p, d1=2p+1.
 * Query is broadcast as (q[d0], q[d1]) per pair → set1_epi32 of those two i16.
 * One madd per pair sums each vector's two squared diffs; accumulate over 8 pairs. */
static inline void group_dists(const int16_t *q, const int16_t *grp, uint32_t out[8]) {
#ifdef __AVX2__
    __m256i acc = _mm256_setzero_si256();
    const int32_t *qp = (const int32_t *)q; /* read pairs of i16 as i32 */
    for (int p = 0; p < 8; p++) {
        __m256i qv = _mm256_set1_epi32(qp[p]);
        __m256i vv = _mm256_loadu_si256((const __m256i *)(grp + p * 16));
        __m256i df = _mm256_sub_epi16(vv, qv);
        __m256i sq = _mm256_madd_epi16(df, df);
        acc = _mm256_add_epi32(acc, sq);
    }
    _mm256_storeu_si256((__m256i *)out, acc);
#else
    /* Scalar fallback for QEMU and other non-AVX2 hosts. */
    for (int k = 0; k < 8; k++) out[k] = 0;
    for (int p = 0; p < 8; p++) {
        int d0 = 2 * p, d1 = 2 * p + 1;
        int32_t q0 = q[d0], q1 = q[d1];
        const int16_t *pg = grp + p * 16;
        for (int k = 0; k < 8; k++) {
            int32_t a = (int32_t)pg[k * 2 + 0] - q0;
            int32_t b = (int32_t)pg[k * 2 + 1] - q1;
            out[k] += (uint32_t)(a * a + b * b);
        }
    }
#endif
}

/* Scan a single cluster's groups, update top-K nearest. */
static inline void scan_cluster(const int16_t *q, uint32_t cluster,
                                uint32_t best_dist[K], uint8_t best_label[K]) {
    uint32_t n_real = G_real_offsets[cluster + 1] - G_real_offsets[cluster];
    if (n_real == 0) return;
    uint32_t gs = G_group_offsets[cluster];
    uint32_t ge = G_group_offsets[cluster + 1];
    uint32_t last_valid = ((n_real - 1) & 7) + 1;

    for (uint32_t g = gs; g < ge; g++) {
        const int16_t *grp = G_groups + (size_t)g * 128;
        const uint8_t *lbl = G_labels + (size_t)g * 8;
        uint32_t valid = (g + 1 == ge) ? last_valid : 8;
        uint32_t dists[8];
        group_dists(q, grp, dists);
        uint32_t thr = best_dist[K - 1];
        for (uint32_t k = 0; k < valid; k++) {
            uint32_t d = dists[k];
            if (d >= thr) continue;
            int pos = K - 1;
            while (pos > 0 && best_dist[pos - 1] > d) {
                best_dist[pos]  = best_dist[pos - 1];
                best_label[pos] = best_label[pos - 1];
                pos--;
            }
            best_dist[pos]  = d;
            best_label[pos] = lbl[k];
            thr = best_dist[K - 1];
        }
    }
}

/* Pick top nprobe centroids via a single SoA pass with early-exit. */
static void top_centroids(const int16_t q[STRIDE], int nprobe,
                          uint32_t *out_dist, uint32_t *out_idx) {
    for (int i = 0; i < nprobe; i++) out_dist[i] = UINT32_MAX;
    uint32_t n_cgrp = G_nclusters / 8;
    for (uint32_t g = 0; g < n_cgrp; g++) {
        uint32_t d8[8];
        group_dists(q, G_centroid_groups + (size_t)g * 128, d8);
        uint32_t thr = out_dist[nprobe - 1];
        uint32_t min8 = d8[0];
        for (int k = 1; k < 8; k++) if (d8[k] < min8) min8 = d8[k];
        if (min8 >= thr) continue;
        for (int k = 0; k < 8; k++) {
            uint32_t d = d8[k];
            if (d >= thr) continue;
            int pos = nprobe - 1;
            while (pos > 0 && out_dist[pos - 1] > d) {
                out_dist[pos] = out_dist[pos - 1];
                out_idx[pos]  = out_idx[pos - 1];
                pos--;
            }
            out_dist[pos] = d;
            out_idx[pos]  = g * 8 + k;
            thr = out_dist[nprobe - 1];
        }
    }
}

/* IVF search with a runtime-chosen nprobe.
 * 1. Pick nprobe nearest centroids (SoA scan + insertion sort).
 * 2. Scan those clusters, track top-K. Bbox lower-bound prunes per cluster. */
static int search_with_nprobe(const int16_t q[STRIDE], int nprobe) {
    uint32_t best_cdist[NPROBE_REPAIR];
    uint32_t best_cidx[NPROBE_REPAIR];
    top_centroids(q, nprobe, best_cdist, best_cidx);

    uint32_t best_dist[K];
    uint8_t  best_label[K];
    for (int i = 0; i < K; i++) { best_dist[i] = UINT32_MAX; best_label[i] = 0; }

    for (int p = 0; p < nprobe; p++) {
        if (best_cdist[p] == UINT32_MAX) break;
        uint32_t c = best_cidx[p];
        if (bbox_lb(q, c) >= best_dist[K - 1]) continue;
        scan_cluster(q, c, best_dist, best_label);
    }

    int frauds = 0;
    for (int i = 0; i < K; i++) frauds += best_label[i];
    return frauds;
}

static int search_fraud_count(const int16_t q[STRIDE]) {
    int frauds = search_with_nprobe(q, NPROBE_FAST);
    /* Boundary case — re-scan a wider nprobe to fix possible IVF misses. */
    if (frauds == 2 || frauds == 3) {
        frauds = search_with_nprobe(q, NPROBE_REPAIR);
    }
    return frauds;
}

/* ---------- HTTP server ---------- */

typedef struct {
    int fd;
    int is_control;  /* 1 = LB control connection (SCM_RIGHTS), 0 = HTTP client */
    char buf[MAX_BODY + 1024];
    int blen;
} Conn;

static Conn *G_conns;

static void conn_close(int epfd, Conn *c) {
    epoll_ctl(epfd, EPOLL_CTL_DEL, c->fd, NULL);
    close(c->fd);
    c->fd = -1;
    c->blen = 0;
    c->is_control = 0;
}

/* Receive one file descriptor sent via sendmsg(SCM_RIGHTS) on a Unix socket.
 * Returns the received fd, -1 on error, -2 if the peer closed the channel. */
static int recv_fd(int control_fd) {
    char buf[1];
    struct iovec iov = { .iov_base = buf, .iov_len = sizeof(buf) };
    char cbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr msg = {0};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);
    ssize_t n = recvmsg(control_fd, &msg, 0);
    if (n == 0) return -2;
    if (n < 0) return -1;
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    if (!cmsg || cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) {
        return -1;
    }
    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
    return fd;
}

static int send_all(int fd, const char *p, int len) {
    while (len > 0) {
        int n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p += n;
        len -= n;
    }
    return 0;
}

/* Process whatever we have in the buffer. Returns 0 ok, -1 close. */
static int process_buffer(Conn *c) {
    while (c->blen > 0) {
        /* Need full request: headers ending in \r\n\r\n, then optional body of Content-Length. */
        char *hend = memmem(c->buf, c->blen, "\r\n\r\n", 4);
        if (!hend) return 0;
        int hlen = (int)(hend - c->buf) + 4;

        /* Parse method + path. */
        if (c->blen >= 9 && memcmp(c->buf, "GET /read", 9) == 0) {
            if (send_all(c->fd, READY_RESP, READY_RESP_LEN) < 0) return -1;
            /* Consume request (no body). */
            int consumed = hlen;
            if (consumed < c->blen) memmove(c->buf, c->buf + consumed, c->blen - consumed);
            c->blen -= consumed;
            continue;
        }
        if (c->blen >= 17 && memcmp(c->buf, "POST /fraud-score", 17) == 0) {
            /* Find Content-Length. */
            int clen = 0;
            const char *cl = memmem(c->buf, hlen, "Content-Length:", 15);
            if (!cl) cl = memmem(c->buf, hlen, "content-length:", 15);
            if (cl) {
                cl += 15;
                while (*cl == ' ') cl++;
                clen = atoi(cl);
            }
            if (clen > MAX_BODY) return -1;
            if (c->blen < hlen + clen) return 0; /* need more */

            const char *body = c->buf + hlen;
            int16_t q[STRIDE];
            vectorize(body, clen, q);
            int frauds = search_fraud_count(q);
            if (frauds < 0) frauds = 0;
            if (frauds > 5) frauds = 5;
            if (send_all(c->fd, RESP[frauds], RESP_LEN[frauds]) < 0) return -1;

            int consumed = hlen + clen;
            if (consumed < c->blen) memmove(c->buf, c->buf + consumed, c->blen - consumed);
            c->blen -= consumed;
            continue;
        }
        /* Unknown method/path. */
        send_all(c->fd, BAD_RESP, BAD_RESP_LEN);
        return -1;
    }
    return 0;
}

/* ---------- Index loading ---------- */

static int load_index(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open index"); return -1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }
    void *map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); close(fd); return -1; }
    close(fd);

    const char *p = map;
    if (memcmp(p, "RINHA005", 8) != 0) { fprintf(stderr, "bad magic\n"); return -1; }
    uint32_t count, ncl, ngrp;
    memcpy(&count, p + 8, 4);
    memcpy(&ncl,   p + 12, 4);
    memcpy(&ngrp,  p + 16, 4);
    G_count = count;
    G_nclusters = ncl;
    G_ngroups = ngrp;
    if (ncl > MAX_CLUSTERS) {
        fprintf(stderr, "n_clusters %u exceeds MAX_CLUSTERS %d\n", ncl, MAX_CLUSTERS);
        return -1;
    }

    size_t off = 24; /* 8 magic + 4 count + 4 ncl + 4 ngrp + 4 pad */
    G_centroid_groups = (const int16_t  *)(p + off);
    off += (size_t)(ncl / 8) * 128 * sizeof(int16_t);
    G_bbox_min      = (const int16_t  *)(p + off);
    off += (size_t)ncl * VEC_DIMS * sizeof(int16_t);
    G_bbox_max      = (const int16_t  *)(p + off);
    off += (size_t)ncl * VEC_DIMS * sizeof(int16_t);
    G_group_offsets = (const uint32_t *)(p + off);
    off += (size_t)(ncl + 1) * sizeof(uint32_t);
    G_real_offsets  = (const uint32_t *)(p + off);
    off += (size_t)(ncl + 1) * sizeof(uint32_t);
    G_groups        = (const int16_t  *)(p + off);
    off += (size_t)ngrp * 128 * sizeof(int16_t);
    G_labels        = (const uint8_t  *)(p + off);

    /* Lock pages in RAM and ask for transparent hugepages to keep latency
     * stable. Failures are non-fatal (e.g. memlock rlimit too low). */
    if (mlock(map, st.st_size) < 0) {
        fprintf(stderr, "warning: mlock failed: %s\n", strerror(errno));
    }
    madvise(map, st.st_size, MADV_HUGEPAGE);
    madvise(map, st.st_size, MADV_RANDOM);

    fprintf(stderr, "loaded %u vectors in %u clusters (%u groups)\n", count, ncl, ngrp);
    return 0;
}

/* ---------- PGO training driver ----------
 * Loads the index, parses example-payloads.json into a flat array of body
 * substrings, then loops vectorize+search_fraud_count many times so the
 * profile counters reflect the real query distribution. */
static int pgo_train_main(const char *index_path, const char *payloads_path) {
    for (int i = 0; i < 6; i++) RESP_LEN[i] = (int)strlen(RESP[i]);
    if (load_index(index_path) < 0) return 1;

    int fd = open(payloads_path, O_RDONLY);
    if (fd < 0) { perror("open payloads"); return 1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat payloads"); close(fd); return 1; }
    char *json = malloc((size_t)st.st_size + 1);
    if (!json) { close(fd); return 1; }
    if (read(fd, json, st.st_size) != st.st_size) { close(fd); return 1; }
    json[st.st_size] = '\0';
    close(fd);

    /* Slice top-level array into body substrings. */
    int cap = 1024, n = 0;
    char **bodies = malloc((size_t)cap * sizeof(char *));
    int  *blens   = malloc((size_t)cap * sizeof(int));
    int depth = 0;
    char *start = NULL;
    for (long i = 0; i < st.st_size; i++) {
        char c = json[i];
        if (c == '{') {
            if (depth == 0) start = json + i;
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0 && start) {
                if (n >= cap) {
                    cap *= 2;
                    bodies = realloc(bodies, (size_t)cap * sizeof(char *));
                    blens  = realloc(blens,  (size_t)cap * sizeof(int));
                }
                bodies[n] = start;
                blens[n]  = (int)((json + i + 1) - start);
                n++;
                start = NULL;
            }
        }
    }
    fprintf(stderr, "pgo_train: %d payloads, looping...\n", n);

    int iters = 200;
    long total = 0, frauds_total = 0;
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < n; b++) {
            int16_t q[STRIDE];
            vectorize(bodies[b], blens[b], q);
            int f = search_fraud_count(q);
            frauds_total += f;
            total++;
        }
    }
    fprintf(stderr, "pgo_train: %ld queries, frauds_total=%ld\n", total, frauds_total);
    free(bodies); free(blens); free(json);
    return 0;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    if (argc == 4 && strcmp(argv[1], "--pgo-train") == 0) {
        return pgo_train_main(argv[2], argv[3]);
    }
    if (argc != 3) {
        fprintf(stderr, "usage: %s <index.bin> <control_sock_path>\n", argv[0]);
        fprintf(stderr, "       %s --pgo-train <index.bin> <payloads.json>\n", argv[0]);
        return 1;
    }
    signal(SIGPIPE, SIG_IGN);
    for (int i = 0; i < 6; i++) RESP_LEN[i] = (int)strlen(RESP[i]);
    READY_RESP_LEN = (int)strlen(READY_RESP);
    BAD_RESP_LEN = (int)strlen(BAD_RESP);

    if (load_index(argv[1]) < 0) return 1;

    /* Listen on a Unix SOCK_STREAM at the given path. The LB connects to this
     * socket once at startup and uses sendmsg(SCM_RIGHTS) to hand off client
     * file descriptors. */
    const char *sock_path = argv[2];
    unlink(sock_path);
    int srv = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
    if (srv < 0) { perror("socket"); return 1; }
    struct sockaddr_un uaddr = {0};
    uaddr.sun_family = AF_UNIX;
    strncpy(uaddr.sun_path, sock_path, sizeof(uaddr.sun_path) - 1);
    if (bind(srv, (struct sockaddr *)&uaddr, sizeof(uaddr)) < 0) { perror("bind"); return 1; }
    if (listen(srv, 16) < 0) { perror("listen"); return 1; }
    chmod(sock_path, 0666);
    int one = 1;

    int epfd = epoll_create1(EPOLL_CLOEXEC);
    struct epoll_event ev = { .events = EPOLLIN, .data.fd = srv };
    epoll_ctl(epfd, EPOLL_CTL_ADD, srv, &ev);

    G_conns = calloc(MAX_CONNS, sizeof(Conn));
    for (int i = 0; i < MAX_CONNS; i++) G_conns[i].fd = -1;

    struct epoll_event events[EVBATCH];
    fprintf(stderr, "api: listening on %s\n", sock_path);

    while (1) {
        int n = epoll_wait(epfd, events, EVBATCH, -1);
        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;

            if (fd == srv) {
                /* New control connection from the LB. */
                while (1) {
                    int c = accept4(srv, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
                    if (c < 0) break;
                    if (c >= MAX_CONNS) { close(c); continue; }
                    G_conns[c].fd = c;
                    G_conns[c].is_control = 1;
                    G_conns[c].blen = 0;
                    struct epoll_event cev = { .events = EPOLLIN, .data.fd = c };
                    epoll_ctl(epfd, EPOLL_CTL_ADD, c, &cev);
                }
                continue;
            }

            Conn *cn = &G_conns[fd];

            if (cn->is_control) {
                /* LB sent us at least one client fd. Drain. */
                while (1) {
                    int cfd = recv_fd(fd);
                    if (cfd == -2) { conn_close(epfd, cn); break; }
                    if (cfd == -1) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                        conn_close(epfd, cn);
                        break;
                    }
                    int flags = fcntl(cfd, F_GETFL, 0);
                    fcntl(cfd, F_SETFL, flags | O_NONBLOCK);
                    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                    if (cfd >= MAX_CONNS) { close(cfd); continue; }
                    G_conns[cfd].fd = cfd;
                    G_conns[cfd].is_control = 0;
                    G_conns[cfd].blen = 0;
                    struct epoll_event cev = { .events = EPOLLIN, .data.fd = cfd };
                    epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &cev);
                }
                continue;
            }

            /* Client connection: read HTTP, respond. */
            while (1) {
                int space = (int)sizeof(cn->buf) - cn->blen;
                if (space <= 0) { conn_close(epfd, cn); break; }
                int r = recv(fd, cn->buf + cn->blen, space, 0);
                if (r == 0) { conn_close(epfd, cn); break; }
                if (r < 0) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        if (process_buffer(cn) < 0) conn_close(epfd, cn);
                        break;
                    }
                    conn_close(epfd, cn);
                    break;
                }
                cn->blen += r;
                if (process_buffer(cn) < 0) { conn_close(epfd, cn); break; }
            }
        }
    }
    return 0;
}
