// ============================================================================
// platform_net.h — minimal cross-platform socket abstraction
// ----------------------------------------------------------------------------
// Used by fused_server_win.cpp. Provides Windows (Winsock2) / POSIX (BSD socket)
// compatibility for the small surface we actually need:
//   - socket_t                 : SOCKET on Windows, int on POSIX
//   - INVALID_SOCKET_VALUE     : platform-correct invalid handle sentinel
//   - socklen_t_compat         : socklen_t (POSIX) / int (Windows)
//   - ssize_t                  : already in <unistd.h> on POSIX; typedef'd here on Windows
//   - sock_read / sock_write   : recv/send wrappers returning ssize_t
//   - closesocket_compat(fd)   : closesocket on Windows, ::close on POSIX
//   - WinsockInit              : RAII guard around WSAStartup/WSACleanup; no-op on POSIX
//   - disable_sigpipe()        : ignore SIGPIPE on POSIX so writes to dead sockets
//                                don't kill the process; no-op on Windows
//
// Network byte-order helpers (htonl/ntohl/htons/ntohs) are provided by the
// underlying winsock2.h / arpa/inet.h, so we don't redefine them.
// ============================================================================

#pragma once

#ifdef _WIN32

  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif

  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "Ws2_32.lib")

  #include <cstdio>
  #include <cstdint>
  #include <cstdlib>     // std::exit
  #include <basetsd.h>   // SSIZE_T

  using socket_t          = SOCKET;
  using socklen_t_compat  = int;

  // Windows lacks POSIX ssize_t in <sys/types.h>; SSIZE_T from basetsd.h is the equivalent.
  #ifndef _SSIZE_T_DEFINED
    typedef SSIZE_T ssize_t;
    #define _SSIZE_T_DEFINED
  #endif

  static constexpr socket_t INVALID_SOCKET_VALUE = INVALID_SOCKET;

  inline ssize_t sock_read(socket_t fd, void * buf, size_t n) {
      // recv on Windows takes int length; clamp at INT_MAX defensively.
      int to_read = (n > 0x7fffffffu) ? 0x7fffffff : (int) n;
      int r = recv(fd, (char *) buf, to_read, 0);
      return (ssize_t) r;   // SOCKET_ERROR == -1, which propagates as ssize_t -1
  }

  inline ssize_t sock_write(socket_t fd, const void * buf, size_t n) {
      int to_write = (n > 0x7fffffffu) ? 0x7fffffff : (int) n;
      int w = send(fd, (const char *) buf, to_write, 0);
      return (ssize_t) w;
  }

  inline int closesocket_compat(socket_t fd) {
      return ::closesocket(fd);
  }

  // RAII guard around WSAStartup / WSACleanup. Construct once at top of main().
  class WinsockInit {
  public:
      WinsockInit() {
          WSADATA wsa;
          int rc = WSAStartup(MAKEWORD(2, 2), &wsa);
          if (rc != 0) {
              std::fprintf(stderr, "WSAStartup failed: %d\n", rc);
              std::exit(1);
          }
      }
      ~WinsockInit() { WSACleanup(); }
      WinsockInit(const WinsockInit &)             = delete;
      WinsockInit & operator=(const WinsockInit &) = delete;
  };

  inline void disable_sigpipe() { /* no-op on Windows */ }

#else  // POSIX

  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <sys/socket.h>
  #include <sys/types.h>
  #include <unistd.h>
  #include <signal.h>

  using socket_t          = int;
  using socklen_t_compat  = socklen_t;

  static constexpr socket_t INVALID_SOCKET_VALUE = -1;

  inline ssize_t sock_read(socket_t fd, void * buf, size_t n) {
      return ::read(fd, buf, n);
  }

  inline ssize_t sock_write(socket_t fd, const void * buf, size_t n) {
      return ::write(fd, buf, n);
  }

  inline int closesocket_compat(socket_t fd) {
      return ::close(fd);
  }

  class WinsockInit {
  public:
      WinsockInit()  = default;
      ~WinsockInit() = default;
  };

  inline void disable_sigpipe() {
      signal(SIGPIPE, SIG_IGN);
  }

#endif