/*
 * Hilltops Game Server (single-threaded, POSIX sockets)
 * -----------------------------------------------------
 * Usage: ./hilltops_server <port> <matrix_file>
 * Protocol (plain text):
 *   On connect:  Server -> Client: "CONNECTED Hilltops Server\n"
 *   Client -> Server: "PLAY <name>\n"
 *   Server -> Client: rows cols\n
 *                      <rows lines, space-separated ints>\n
 *   Server -> Client: "SEND_SWAPS\n"
 *   Client -> Server: N\n  (number of swaps)
 *                      <N lines, each: x1 y1 x2 y2>\n
 *   Server -> Client: if hilltop-perfect after applying swaps:
 *                        "OK <N> <max_worst_distance>\n"
 *                     else
 *                        "NOT_OK\n"
 *   Server -> Client (on end of request): "DISCONNECTED\n"
 *
 * Additional command:
 *   Client -> Server: "RESULTS\n"  (server responds with a sorted leaderboard, then sends "DISCONNECTED\n" and shuts down)
 *
 * Session lifecycle:
 *   - On "PLAY <name>": run one game, reply OK/NOT_OK, then send DISCONNECTED and close the client connection; server continues.
 *   - On "RESULTS": send leaderboard, send DISCONNECTED, then shut down the server after disconnecting the client.
 *
 * Logging:
 *   Per-connection log file: session_<pid>_<conn_idx>_<player>.txt
 *   Includes: initial matrix, swaps, matrix **after every swap**, final matrix, worst distances & reachability matrices, verdict.
 *
 * Notes:
 *   - Simple, blocking, one client at a time. Incoming connections queue in backlog.
 *   - Uses your HeightsMatrix.h as-is for swaps & hilltop check.
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "heightsmatrix.h" // Use the provided header as-is

static volatile sig_atomic_t g_should_exit = 0;
static void handle_sigint(int) { g_should_exit = 1; }

// ---------------------------- small helpers --------------------------------

static bool send_all(int fd, const std::string &data) {
    const char *buf = data.c_str();
    size_t left = data.size();
    while (left > 0) {
        ssize_t n = ::send(fd, buf, left, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        buf += n;
        left -= (size_t)n;
    }
    return true;
}

static std::optional<std::string> recv_line(int fd) {
    std::string line;
    char ch;
    while (true) {
        ssize_t n = ::recv(fd, &ch, 1, 0);
        if (n == 0) {
            if (line.empty()) return std::nullopt; // peer closed
            break; // return what's collected (no trailing \n)
        }
        if (n < 0) {
            if (errno == EINTR) continue;
            return std::nullopt;
        }
        if (ch == '\r') continue; // tolerate CRLF
        if (ch == '\n') break;
        line.push_back(ch);
    }
    return line;
}

static void print_matrix(std::ostream &os, const std::vector<std::vector<int>> &m) {
    for (const auto &row : m) {
        for (size_t j = 0; j < row.size(); ++j) {
            os << std::setw(3) << row[j]; 
	    os << ' ';
        }
	os << '\n';
    }
}

static std::optional<std::tuple<int,int,std::vector<std::vector<int>>>> load_matrix(const std::string &path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;
    int R, C;
    if (!(in >> R >> C)) return std::nullopt;
    if (R <= 0 || C <= 0 || R > 20 || C > 20) return std::nullopt;
    std::vector<std::vector<int>> a(R, std::vector<int>(C));
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            if (!(in >> a[i][j])) return std::nullopt;
        }
    }
    return std::make_tuple(R, C, std::move(a));
}

static std::string matrix_to_wire(int R, int C, const std::vector<std::vector<int>> &a) {
    std::ostringstream out;
    out << R << ' ' << C << "\n";
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            out << a[i][j] << (j + 1 == C ? '\n' : ' ');
        }
    }
    return out.str();
}

static std::string sanitize_name(const std::string &name) {
    std::string s;
    for (char ch : name) {
        if (std::isalnum((unsigned char)ch)) s.push_back(ch);
        else if (ch == '-' || ch == '_') s.push_back(ch);
        else s.push_back('_');
    }
    if (s.empty()) s = "player";
    return s;
}

// ---------------------------- results store --------------------------------

struct Result { bool ok; int swaps; int maxWorst; int seq; };
// Global in-memory attempts indexed by player name (simple & fine single-threaded)
static std::map<std::string, std::vector<Result>> g_results;
static int g_seq_counter = 0; // to break ties deterministically

static void record_attempt(const std::string &player, bool ok, int swaps, int maxWorst) {
    g_results[player].push_back(Result{ok, swaps, maxWorst, g_seq_counter++});
}

static std::string format_results_sorted() {
    struct Row { std::string name; bool ok; int swaps; int maxWorst; int seq; };
    std::vector<Row> rows;
    rows.reserve(64);
    for (const auto &kv : g_results) {
        const std::string &name = kv.first;
        for (const auto &r : kv.second) rows.push_back(Row{name, r.ok, r.swaps, r.maxWorst, r.seq});
    }
    auto cmp = [](const Row &a, const Row &b) {
        if (a.ok != b.ok) return a.ok && !b.ok; // OK first, losses (-1) last
        if (!a.ok && !b.ok) return a.seq < b.seq; // stable-ish order among losses
        if (a.swaps != b.swaps) return a.swaps < b.swaps;
        if (a.maxWorst != b.maxWorst) return a.maxWorst < b.maxWorst;
        if (a.name != b.name) return a.name < b.name;
        return a.seq < b.seq;
    };
    std::sort(rows.begin(), rows.end(), cmp);

    std::ostringstream out;
    out << "RESULTS_BEGIN\n";
    for (const auto &r : rows) {
        if (r.ok) out << r.name << ' ' << r.swaps << ' ' << r.maxWorst << "\n";
        else      out << r.name << " -1\n";
    }
    out << "END\n";
    return out.str();
}

// ------------------------ per-connection handling --------------------------

struct Swap { int x1, y1, x2, y2; };

static bool run_one_game(int cfd, const std::string &player, const std::string &matrix_file, std::ofstream &log) {
    // Load a fresh matrix for this attempt
    auto loaded = load_matrix(matrix_file);
    if (!loaded) { send_all(cfd, "ERROR cannot load matrix file\n"); return false; }
    int R, C; std::vector<std::vector<int>> A;
    std::tie(R, C, A) = *loaded;

    log << "\n=== Player: " << player << " ===\n";
    log << "\n=== Initial Matrix (" << R << 'x' << C << ") ===\n";
    print_matrix(log, A);

    // Send matrix and request swaps
    send_all(cfd, matrix_to_wire(R, C, A));
    send_all(cfd, "SEND_SWAPS\n");

    // Read N
    auto nline = recv_line(cfd);
    if (!nline) { send_all(cfd, "ERROR missing N\n"); return false; }
    int N = 0; {
        std::istringstream iss(*nline);
        if (!(iss >> N) || N < 0) { send_all(cfd, "ERROR bad N\n"); return false; }
    }

    std::vector<Swap> swaps; swaps.reserve((size_t)N);
    for (int i = 0; i < N; ++i) {
        auto sline = recv_line(cfd);
        if (!sline) { send_all(cfd, "ERROR missing swap line\n"); return false; }
        std::istringstream iss(*sline);
        Swap s{}; if (!(iss >> s.x1 >> s.y1 >> s.x2 >> s.y2)) { send_all(cfd, "ERROR malformed swap\n"); return false; }
        swaps.push_back(s);
    }

    HeightsMatrix H(std::move(A), R, C);
    // Apply swaps and log matrix after each
    for (int i = 0; i < N; ++i) {
        const auto &s = swaps[i];
        if (!H.swapCells(s.x1, s.y1, s.x2, s.y2)) {
            send_all(cfd, "ERROR invalid swap indices\n");
            log << "Swap failed due to invalid indices.\n";
            record_attempt(player, /*ok=*/false, /*swaps=*/-1, /*maxWorst=*/-1);
            return false;
        }
        log << "--- After swap " << (i+1) << " : " << "(" << swaps[i].x1 << ',' << swaps[i].y1 << ") <-> (" << swaps[i].x2 << ',' << swaps[i].y2 << ")"  << " ---\n";
        {
            std::ostringstream oss; std::streambuf *old = std::cout.rdbuf(oss.rdbuf()); H.print(); std::cout.rdbuf(old); log << oss.str();
        }
    }

    auto [ok, maxWorst] = H.isHilltopPerfect();

    log << "\n=== Final Matrix ===\n";
    {
        std::ostringstream oss; std::streambuf *old = std::cout.rdbuf(oss.rdbuf()); H.print(); std::cout.rdbuf(old); log << oss.str();
    }
	
    log << "\n=== Reachability ===\n";
    {
        std::ostringstream oss; std::streambuf *old = std::cout.rdbuf(oss.rdbuf()); H.printreachable(); std::cout.rdbuf(old); log << oss.str();
    }

    log << "\n=== Worst Distances ===\n";
    {
        std::ostringstream oss; std::streambuf *old = std::cout.rdbuf(oss.rdbuf()); H.printworstDistance(); std::cout.rdbuf(old); log << oss.str();
    }

    log << "\n";

    if (ok) {
        std::ostringstream out; out << "OK " << N << ' ' << maxWorst << "\n"; send_all(cfd, out.str());
        log << "=== Verdict ===\nOK\nNumber of swaps: " << N << "\nMax worst distance: " << maxWorst << "\n";
        record_attempt(player, /*ok=*/true, N, maxWorst);
    } else {
        send_all(cfd, "NOT_OK\n");
        log << "=== Verdict ===\nNOT_OK\n";
        record_attempt(player, /*ok=*/false, /*swaps=*/-1, /*maxWorst=*/-1);
    }

    log.flush();
    return true;
}

enum class ClientOutcome { Continue, Shutdown };

static ClientOutcome handle_client(int cfd, int conn_idx, const std::string &matrix_file) {
    // On connect greeting
    send_all(cfd, "CONNECTED Hilltops Server\n");

    std::string player; // set after PLAY <name>
    std::unique_ptr<std::ofstream> plog; // lazily opened on first PLAY

    while (true) {
        auto line = recv_line(cfd);
        if (!line) return ClientOutcome::Continue; // peer closed

        // Trim leading spaces
        while (!line->empty() && std::isspace((unsigned char)line->front())) line->erase(line->begin());

        if (line->rfind("PLAY ", 0) == 0) {
            player = line->substr(5);
            if (player.empty()) player = "player";
            std::string sname = sanitize_name(player);
            if (!plog) {
                std::ostringstream logname; logname << "session_" << ::getpid() << '_' << conn_idx << '_' << sname << ".txt";
                plog = std::make_unique<std::ofstream>(logname.str());
                (*plog) << "# Log for player: " << player << "\n";
            }
            // Run one game attempt
            run_one_game(cfd, player, matrix_file, *plog);
            // Notify disconnection and return
            send_all(cfd, "DISCONNECTED\n");
            return ClientOutcome::Continue;
        } else if (*line == "RESULTS") {
            auto body = format_results_sorted();
            send_all(cfd, body);
            // Notify and shut down per request
            send_all(cfd, "DISCONNECTED\n");
            return ClientOutcome::Shutdown;
        } else {
            send_all(cfd, "ERROR expected PLAY <name> | RESULTS\n");
        }
    }
}

// -------------------------------- main -------------------------------------

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <port> <matrix_file>\n";
        return 1;
    }

    int port = std::stoi(argv[1]);
    std::string matrix_file = argv[2];

    std::signal(SIGINT, handle_sigint);

    int sfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) { perror("socket"); return 1; }

    int yes = 1;
    if (setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) < 0) {
        perror("setsockopt");
    }

    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons((uint16_t)port);
    if (bind(sfd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); ::close(sfd); return 1; }

    if (listen(sfd, 8) < 0) { perror("listen"); ::close(sfd); return 1; }

    std::cout << "Hilltops server listening on port " << port << "... (Ctrl+C to stop)\n";

    int conn_idx = 0;
    while (!g_should_exit) {
        sockaddr_in cli{}; socklen_t cl = sizeof(cli);
        int cfd = ::accept(sfd, (sockaddr*)&cli, &cl);
        if (cfd < 0) {
            if (errno == EINTR) break;
            perror("accept");
            continue;
        }
        ++conn_idx;
        char ipbuf[64] = {0};
        inet_ntop(AF_INET, &cli.sin_addr, ipbuf, sizeof(ipbuf));
        std::cout << "Client " << conn_idx << " connected from " << ipbuf << ":" << ntohs(cli.sin_port) << "\n";

        auto outcome = handle_client(cfd, conn_idx, matrix_file);
        ::close(cfd);
        std::cout << "Client " << conn_idx << " disconnected.\n";
        if (outcome == ClientOutcome::Shutdown) {
            std::cout << "Shutting down per RESULTS command.\n";
            break;
        }
    }

    ::close(sfd);
    std::cout << "Server stopped.\n";
    return 0;
}
