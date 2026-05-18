/*
 * lb: minimal TCP load balancer that hands each accepted client socket to one
 * of two API instances via SCM_RIGHTS over a Unix control socket. The LB never
 * reads or writes the HTTP bytes — it just round-robins the file descriptors.
 *
 * Topology:
 *   client --TCP:9999--> [lb] --recvmsg(SCM_RIGHTS) over UNIX--> [api1|api2]
 *
 * APIs listen on /sockets/api{1,2}.sock (Unix SOCK_STREAM). The LB connects
 * once to each at startup and reuses both control connections forever.
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
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>

#define N_APIS 2

static int connect_unix(const char *path) {
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    /* APIs may not be listening yet at startup — retry briefly. */
    for (int i = 0; i < 100; i++) {
        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) == 0) return fd;
        usleep(100 * 1000);
    }
    close(fd);
    return -1;
}

static int send_fd(int control_fd, int client_fd) {
    char buf[1] = {0};
    struct iovec iov = { .iov_base = buf, .iov_len = 1 };
    char cbuf[CMSG_SPACE(sizeof(int))];
    struct msghdr msg = {0};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cbuf;
    msg.msg_controllen = sizeof(cbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &client_fd, sizeof(int));
    return sendmsg(control_fd, &msg, MSG_NOSIGNAL);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <port> <api1.sock> <api2.sock>\n", argv[0]);
        return 1;
    }
    signal(SIGPIPE, SIG_IGN);
    int port = atoi(argv[1]);

    int ctl[N_APIS];
    ctl[0] = connect_unix(argv[2]);
    ctl[1] = connect_unix(argv[3]);
    if (ctl[0] < 0 || ctl[1] < 0) {
        fprintf(stderr, "lb: connect_unix failed\n");
        return 1;
    }
    fprintf(stderr, "lb: connected to %s and %s\n", argv[2], argv[3]);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 1; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(srv, 4096) < 0) { perror("listen"); return 1; }
    fprintf(stderr, "lb: listening on :%d\n", port);

    int next = 0;
    while (1) {
        int c = accept(srv, NULL, NULL);
        if (c < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        int api = next;
        next = (next + 1) % N_APIS;
        if (send_fd(ctl[api], c) < 0) {
            /* Try the other backend once. */
            int other = (api + 1) % N_APIS;
            if (send_fd(ctl[other], c) < 0) {
                fprintf(stderr, "lb: both apis unreachable, dropping client\n");
            }
        }
        close(c); /* LB no longer needs the fd; the API has its own copy. */
    }
    return 0;
}
