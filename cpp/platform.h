// platform.h
//
// Cross-platform shim for SimpleT2M. Hides win32 vs posix socket API
// differences and the ORT wide-char path quirk on Windows.
//
// Include this BEFORE <onnxruntime_cxx_api.h>.

#pragma once

// ============================================================
// Sockets
// ============================================================
#ifdef _WIN32
  // Must come BEFORE <windows.h>:
  //   NOMINMAX            — suppress the min()/max() macros that break std::min/std::max
  //   WIN32_LEAN_AND_MEAN — trim windows.h, also avoids further macro pollution
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif

  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
  #pragma comment(lib, "ws2_32.lib")

  #include <stdexcept>   // for std::runtime_error used below
  #include <stdexcept>

  // Windows lacks the POSIX ssize_t. npz_reader.h uses it.
  // SSIZE_T (from <basetsd.h>, pulled in by windows.h) is the equivalent.
  // Guard with _SSIZE_T_DEFINED so this stays consistent with platform_net.h,
  // which defines the same typedef — avoids a redefinition if both are included.
  #include <basetsd.h>
  #ifndef _SSIZE_T_DEFINED
    typedef SSIZE_T ssize_t;
    #define _SSIZE_T_DEFINED
  #endif

  using sock_t = SOCKET;
  static constexpr sock_t INVALID_SOCK = INVALID_SOCKET;

  inline int close_sock(sock_t s) { return ::closesocket(s); }
  inline int last_sock_err() { return WSAGetLastError(); }

  struct PlatformNetInit {
      PlatformNetInit() {
          WSADATA d;
          if (WSAStartup(MAKEWORD(2,2), &d) != 0)
              throw std::runtime_error("WSAStartup failed");
      }
      ~PlatformNetInit() { WSACleanup(); }
  };

  // ORT on Windows wants wchar_t paths
  #include <string>
  inline std::wstring ort_path(const std::string& s) {
      int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
      std::wstring w(n, 0);
      MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
      if (!w.empty() && w.back() == 0) w.pop_back();
      return w;
  }

#else  // POSIX (macOS / Linux)
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <unistd.h>
  #include <errno.h>
  #include <cstring>
  #include <string>

  using sock_t = int;
  static constexpr sock_t INVALID_SOCK = -1;

  // POSIX setsockopt takes void*, but we keep a uniform call site
  #ifndef SOCKET_ERROR
    #define SOCKET_ERROR (-1)
  #endif

  inline int close_sock(sock_t s) { return ::close(s); }
  inline int last_sock_err() { return errno; }

  // No-op on POSIX; struct exists so call sites compile cross-platform
  struct PlatformNetInit {
      PlatformNetInit() = default;
      ~PlatformNetInit() = default;
  };

  // ORT on macOS/Linux accepts UTF-8 const char* directly. We return
  // std::string here so the call site is `ort_path(p).c_str()`, matching
  // the Windows wchar_t* pattern.
  inline std::string ort_path(const std::string& s) { return s; }
#endif
