// Private-VM positive control only. Never runs a model or opens a camera.
// No file trace, payload retention, arbitrary destinations or host bypass.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#include <bcrypt.h>
#include "afd_schema.h"
#include <atomic>
#include <cstdlib>
#include <limits>
#include <mutex>

namespace {
constexpr char private_uuid[] = "2cb2eaf1-068d-47a3-8539-8b0d9084b32e";
constexpr GUID private_identity = {0x2cb2eaf1, 0x068d, 0x47a3,
    {0x85, 0x39, 0x8b, 0x0d, 0x90, 0x84, 0xb3, 0x2e}};
constexpr GUID trace_metadata = {0x68fdd900, 0x4a3e, 0x11d1,
    {0x84, 0xf4, 0x00, 0x00, 0xf8, 0x04, 0x64, 0xe3}};
constexpr size_t max_events = 50000;
bool owned_trace_unstopped = false;

template<class T> T read(const BYTE* bytes) {
    T value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

LONGLONG now() {
    LARGE_INTEGER value;
    require(QueryPerformanceCounter(&value) != 0 && value.QuadPart > 0, "QPC unavailable");
    return value.QuadPart;
}

void private_machine() {
    // Microsoft uses the little-endian GUID layout in SMBIOS 2.6 and later.
    constexpr DWORD rsmb = 0x52534d42;
    const UINT count = GetSystemFirmwareTable(rsmb, 0, nullptr, 0);
    require(count >= 8 && count <= max_metadata, "Cannot read private-machine SMBIOS");
    std::vector<BYTE> bytes(count);
    require(GetSystemFirmwareTable(rsmb, 0, bytes.data(), count) == count,
            "SMBIOS changed during inspection");
    require(bytes[1] > 2 || (bytes[1] == 2 && bytes[2] >= 6), "Unsupported SMBIOS UUID layout");
    require(read<DWORD>(bytes.data() + 4) == count - 8, "Invalid SMBIOS table bounds");
    size_t at = 8;
    unsigned identities = 0;
    bool ended = false;
    while (at < bytes.size()) {
        require(bytes.size() - at >= 4, "Truncated SMBIOS header");
        const BYTE kind = bytes[at], length = bytes[at + 1];
        require(length >= 4 && at + length <= bytes.size(), "Invalid SMBIOS record bounds");
        size_t end = at + length;
        while (end + 1 < bytes.size() && (bytes[end] != 0 || bytes[end + 1] != 0)) ++end;
        require(end + 1 < bytes.size(), "Unterminated SMBIOS record");
        if (kind == 1) {
            require(length >= 24, "Missing private machine UUID");
            const auto identity = read<GUID>(bytes.data() + at + 8);
            require(IsEqualGUID(identity, private_identity), "Refusing to run outside the exact private VM");
            ++identities;
        }
        at = end + 2;
        if (kind == 127) { ended = true; break; }
    }
    require(ended && at == bytes.size() && identities == 1,
            "Private VM identity is incomplete or ambiguous");

    ULONG size = 0;
    const ULONG first = GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_SKIP_ANYCAST
        | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER, nullptr, nullptr, &size);
    if (first == ERROR_NO_DATA) return;
    require(first == ERROR_BUFFER_OVERFLOW && size >= sizeof(IP_ADAPTER_ADDRESSES)
            && size <= max_metadata, "Cannot inspect guest adapter isolation");
    std::vector<BYTE> adapters(size);
    check(GetAdaptersAddresses(AF_UNSPEC, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST
        | GAA_FLAG_SKIP_DNS_SERVER, nullptr,
        reinterpret_cast<IP_ADAPTER_ADDRESSES*>(adapters.data()), &size), "Inspect guest adapters");
    unsigned seen = 0;
    for (auto* item = reinterpret_cast<IP_ADAPTER_ADDRESSES*>(adapters.data()); item; item = item->Next) {
        require(++seen <= 256, "Unexpected guest adapter count");
        require(item->IfType == IF_TYPE_SOFTWARE_LOOPBACK || item->OperStatus != IfOperStatusUp,
                "Disable VM networking before the isolated observer test");
    }
}

struct Field {
    const char* name;
    unsigned flags, in_type, out_type, length;
};

std::vector<Field> fields(USHORT id) {
    std::vector<Field> result = {{"Process", 0, 16, 19, 8}, {"Endpoint", 0, 16, 19, 8}};
    if (id == 1) {
        result.insert(result.end(), {{"AddressFamily", 0, 8, 8, 4}, {"SocketType", 0, 8, 8, 4},
            {"Protocol", 0, 8, 8, 4}, {"UserModePid", 0, 16, 19, 8}});
    } else if (id == 13 || id == 14) {
        result.push_back({"Error", 0, 8, 31, 4});
    } else {
        if (id == 21 || id == 22) result.insert(result.end(), {{"FastPath", 0, 13, 13, 4},
            {"BufferCount", 0, 7, 7, 4}, {"Buffer", 0, 16, 19, 8}, {"BufferLength", 0, 8, 18, 4}});
        if (id == 5 || id == 22) result.push_back({"Address", 16, 14, 24, 16});
        else result.push_back({"Address", 0, 8, 23, 4});
        result.push_back({"Port", 0, 6, 22, 2});
    }
    return result;
}

std::string expected_schema(USHORT id) {
    const auto properties = fields(id);
    std::ostringstream out;
    out << "{\"event_id\":" << id << ",\"version\":0,\"channel\":16,\"level\":"
        << (id == 21 || id == 22 ? 5 : 4)
        << ",\"opcode\":0,\"task\":0,\"keywords\":9223372036854775808,\"top_level_properties\":"
        << properties.size() << ",\"properties\":[";
    for (size_t i = 0; i < properties.size(); ++i) {
        const auto& field = properties[i];
        if (i) out << ',';
        out << "{\"name\":" << quoted(field.name) << ",\"flags\":" << field.flags
            << ",\"in_type\":" << field.in_type << ",\"out_type\":" << field.out_type
            << ",\"count_or_index\":1,\"length_or_index\":" << field.length << '}';
    }
    out << "]}";
    return out.str();
}

void reviewed_schemas() {
    require(sizeof(void*) == 8, "Observer requires reviewed 64-bit Windows layout");
    const auto all = descriptors();
    for (USHORT id : selected) {
        std::vector<EVENT_DESCRIPTOR> matches;
        for (const auto& descriptor : all)
            if (descriptor.Id == id && descriptor.Version == 0) matches.push_back(descriptor);
        require(matches.size() == 1 && schema(matches.front()) == expected_schema(id),
                "Guest AFD schema differs from the reviewed metadata-only layout");
    }
}

struct Handle {
    HANDLE value = nullptr;
    explicit Handle(HANDLE value_ = nullptr) : value(value_) {}
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

struct Socket {
    SOCKET value = INVALID_SOCKET;
    explicit Socket(SOCKET value_) : value(value_) { require(value != INVALID_SOCKET, "Socket creation failed"); }
    ~Socket() { if (value != INVALID_SOCKET) closesocket(value); }
    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
};

struct Winsock {
    Winsock() {
        WSADATA data;
        require(WSAStartup(MAKEWORD(2, 2), &data) == 0, "Guest Winsock initialization failed");
    }
    ~Winsock() { WSACleanup(); }
};

struct Capture {
    std::mutex mutex;
    std::vector<std::string> events;
    std::atomic<bool> decoder_ok{true}, limit{false};
    std::atomic<ULONG> consumer_status{ERROR_IO_PENDING};
    LONGLONG begin = 0, end = 0, frequency = 0;
};

void WINAPI receive(EVENT_RECORD* record) noexcept {
    auto& data = *static_cast<Capture*>(record->UserContext);
    try {
        // Session bookkeeping is not a network provider and has no app data.
        if (!IsEqualGUID(record->EventHeader.ProviderId, provider)) {
            if (IsEqualGUID(record->EventHeader.ProviderId, trace_metadata)) return;
            throw std::runtime_error("Unexpected provider in isolated trace");
        }
        const auto& descriptor = record->EventHeader.EventDescriptor;
        bool allowed = false;
        for (USHORT id : selected) if (descriptor.Id == id) allowed = true;
        require(allowed && descriptor.Version == 0 && descriptor.Channel == 16
                && descriptor.Task == 0 && descriptor.Opcode == 0
                && descriptor.Keyword == 0x8000000000000000ULL
                && descriptor.Level == (descriptor.Id == 21 || descriptor.Id == 22 ? 5 : 4),
                "Unreviewed event descriptor");
        require((record->EventHeader.Flags & EVENT_HEADER_FLAG_32_BIT_HEADER) == 0,
                "32-bit event pointer layout is not reviewed");
        size_t expected = 0;
        for (const auto& field : fields(descriptor.Id)) expected += field.length;
        require(record->UserData && record->UserDataLength == expected,
                "Unexpected event payload shape; no raw bytes retained");
        const auto* bytes = static_cast<const BYTE*>(record->UserData);
        const auto process = read<ULONGLONG>(bytes), endpoint = read<ULONGLONG>(bytes + 8);
        require(process != 0 && endpoint != 0, "Missing socket identity");
        const auto timestamp = record->EventHeader.TimeStamp.QuadPart;
        require(timestamp >= data.begin && timestamp - data.begin <= 30 * data.frequency,
                "Event timestamp outside bounded QPC capture");
        std::lock_guard<std::mutex> lock(data.mutex);
        if (data.events.size() >= max_events) { data.limit = true; return; }
        std::ostringstream out;
        out << "{\"sequence\":" << data.events.size() << ",\"qpc\":" << timestamp
            << ",\"provider\":\"e53c6823-7bb8-44bb-90dc-3f86090d48a6\",\"event_id\":"
            << descriptor.Id << ",\"version\":0,\"header_pid\":" << record->EventHeader.ProcessId
            << ",\"process_key\":" << process << ",\"endpoint_key\":" << endpoint;
        if (descriptor.Id == 1) {
            const auto type = read<ULONG>(bytes + 20), protocol = read<ULONG>(bytes + 24);
            const auto pid = read<ULONGLONG>(bytes + 28);
            require(type > 0 && type <= 65535 && protocol <= 65535
                    && pid > 0 && pid <= std::numeric_limits<ULONG>::max(), "Invalid socket owner metadata");
            out << ",\"owner_pid\":" << pid << ",\"socket_type\":" << type
                << ",\"protocol\":" << protocol;
        } else out << ",\"owner_pid\":null,\"socket_type\":null,\"protocol\":null";
        if (descriptor.Id == 4 || descriptor.Id == 5 || descriptor.Id == 21 || descriptor.Id == 22) {
            const bool ipv6 = descriptor.Id == 5 || descriptor.Id == 22;
            const size_t address_at = descriptor.Id == 21 || descriptor.Id == 22 ? 36 : 16;
            char address[INET6_ADDRSTRLEN]{};
            require(InetNtopA(ipv6 ? AF_INET6 : AF_INET, bytes + address_at, address,
                    static_cast<DWORD>(sizeof(address))) != nullptr, "Cannot format remote address");
            const auto port = ntohs(read<USHORT>(bytes + address_at + (ipv6 ? 16 : 4)));
            require(port != 0, "Missing remote port");
            out << ",\"remote_address\":" << quoted(address) << ",\"remote_port\":" << port;
        } else out << ",\"remote_address\":null,\"remote_port\":null";
        out << '}';
        data.events.push_back(out.str());
    } catch (...) { data.decoder_ok = false; }
}

struct Trace {
    Capture& data;
    std::vector<BYTE> properties;
    TRACEHANDLE controller = 0, consumer = INVALID_PROCESSTRACE_HANDLE;
    Handle thread;
    bool started = false, stopped = false, enabled = false;
    ULONGLONG lost_events = 0, lost_buffers = 0;
    std::wstring name;

    explicit Trace(Capture& data_) : data(data_), properties(sizeof(EVENT_TRACE_PROPERTIES) + 512, 0) {}
    Trace(const Trace&) = delete;
    Trace& operator=(const Trace&) = delete;
    ~Trace() { stop(); }
    EVENT_TRACE_PROPERTIES* props() { return reinterpret_cast<EVENT_TRACE_PROPERTIES*>(properties.data()); }

    static DWORD WINAPI consume(void* argument) {
        auto& self = *static_cast<Trace*>(argument);
        self.data.consumer_status = ProcessTrace(&self.consumer, 1, nullptr, nullptr);
        return 0;
    }

    void start() {
        auto* p = props();
        p->Wnode.BufferSize = static_cast<ULONG>(properties.size());
        require(BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(&p->Wnode.Guid), sizeof(GUID),
                BCRYPT_USE_SYSTEM_PREFERRED_RNG) >= 0, "Cannot create private session identity");
        p->Wnode.ClientContext = 1;  // QPC timestamps, not wall clock.
        p->Wnode.Flags = WNODE_FLAG_TRACED_GUID;
        p->BufferSize = 64;
        p->MinimumBuffers = 8;
        p->MaximumBuffers = 64;
        p->FlushTimer = 1;
        p->LogFileMode = EVENT_TRACE_REAL_TIME_MODE;
        p->LoggerNameOffset = sizeof(EVENT_TRACE_PROPERTIES);
        name = L"CarlCamPrivateControl-" + std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(data.begin);
        require((name.size() + 1) * sizeof(wchar_t) <= 512, "Trace name exceeds allocated buffer");
        std::memcpy(properties.data() + p->LoggerNameOffset, name.c_str(), (name.size() + 1) * sizeof(wchar_t));
        check(StartTraceW(&controller, name.c_str(), p), "Start guest-only real-time trace");
        started = true;
        owned_trace_unstopped = true;
        EVENT_TRACE_LOGFILEW log{};
        log.LoggerName = name.data();
        log.ProcessTraceMode = PROCESS_TRACE_MODE_REAL_TIME | PROCESS_TRACE_MODE_EVENT_RECORD
            | PROCESS_TRACE_MODE_RAW_TIMESTAMP;
        log.EventRecordCallback = receive;
        log.Context = &data;
        consumer = OpenTraceW(&log);
        require(consumer != INVALID_PROCESSTRACE_HANDLE, "Cannot open guest real-time consumer");
        thread.value = CreateThread(nullptr, 0, consume, this, 0, nullptr);
        require(thread.value != nullptr, "Cannot start guest trace consumer");
        std::array<BYTE, offsetof(EVENT_FILTER_EVENT_ID, Events) + selected.size() * sizeof(USHORT)> filter{};
        auto* ids = reinterpret_cast<EVENT_FILTER_EVENT_ID*>(filter.data());
        ids->FilterIn = TRUE;
        ids->Count = static_cast<USHORT>(selected.size());
        std::memcpy(ids->Events, selected.data(), selected.size() * sizeof(USHORT));
        EVENT_FILTER_DESCRIPTOR descriptor{};
        descriptor.Ptr = reinterpret_cast<ULONGLONG>(filter.data());
        descriptor.Size = static_cast<ULONG>(filter.size());
        descriptor.Type = EVENT_FILTER_TYPE_EVENT_ID;
        ENABLE_TRACE_PARAMETERS parameters{};
        parameters.Version = ENABLE_TRACE_PARAMETERS_VERSION_2;
        parameters.EnableFilterDesc = &descriptor;
        parameters.FilterDescCount = 1;
        check(EnableTraceEx2(controller, &provider, EVENT_CONTROL_CODE_ENABLE_PROVIDER,
                TRACE_LEVEL_VERBOSE, ~0ULL, 0, 2000, &parameters), "Enable bounded guest AFD event filter");
        enabled = true;
    }

    void stop() noexcept {
        bool clean = true;
        if (started && !stopped) {
            // Also disable after a timed-out enable call: its callback may have run.
            if (EnableTraceEx2(controller, &provider, EVENT_CONTROL_CODE_DISABLE_PROVIDER,
                    0, 0, 0, 2000, nullptr) != ERROR_SUCCESS) clean = false;
            enabled = false;
            const ULONG status = ControlTraceW(controller, name.c_str(), props(), EVENT_TRACE_CONTROL_STOP);
            stopped = status == ERROR_SUCCESS;
            if (stopped) owned_trace_unstopped = false;
            clean = clean && stopped;
            lost_events = props()->EventsLost;
            lost_buffers = static_cast<ULONGLONG>(props()->RealTimeBuffersLost) + props()->LogBuffersLost;
        }
        // Let ProcessTrace drain the stopped session before closing its handle.
        // Closing first can discard queued events without an EventsLost count.
        if (thread.value && stopped && WaitForSingleObject(thread.value, 5000) != WAIT_OBJECT_0)
            clean = false;
        if (consumer != INVALID_PROCESSTRACE_HANDLE) {
            const ULONG status = CloseTrace(consumer);
            if (status != ERROR_SUCCESS && status != ERROR_CTX_CLOSE_PENDING) clean = false;
        }
        if (thread.value && WaitForSingleObject(thread.value, 5000) != WAIT_OBJECT_0) {
            // Do not free callback state while a consumer is alive. The owned
            // session has already been stopped; exit only this helper process.
            std::cout << "{\"error\":\"Consumer did not stop; guest session cleanup needs review\","
                "\"cleanup_required\":" << (owned_trace_unstopped ? "true" : "false")
                << ",\"privacy_verified\":false}\n" << std::flush;
            std::_Exit(3);
        }
        consumer = INVALID_PROCESSTRACE_HANDLE;
        if (!clean) data.decoder_ok = false;
    }
};

std::wstring executable_path() {
    std::vector<wchar_t> path(32768);
    const DWORD count = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    require(count > 0 && count < path.size(), "Cannot identify the control executable");
    return {path.data(), count};
}

USHORT port_argument(const wchar_t* value) {
    unsigned port = 0, digits = 0;
    for (const wchar_t* c = value; *c; ++c) {
        require(*c >= L'0' && *c <= L'9' && ++digits <= 5, "Invalid loopback port argument");
        port = port * 10 + static_cast<unsigned>(*c - L'0');
    }
    require(digits != 0 && port > 0 && port <= 65535, "Loopback port out of range");
    return static_cast<USHORT>(port);
}

void destination(bool ipv6, USHORT port, sockaddr_storage& address, int& size) {
    address = {};
    if (ipv6) {
        auto* typed = reinterpret_cast<sockaddr_in6*>(&address);
        typed->sin6_family = AF_INET6;
        typed->sin6_addr = in6addr_loopback;
        typed->sin6_port = htons(port);
        size = sizeof(*typed);
    } else {
        auto* typed = reinterpret_cast<sockaddr_in*>(&address);
        typed->sin_family = AF_INET;
        typed->sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        typed->sin_port = htons(port);
        size = sizeof(*typed);
    }
}

bool known_label(const std::wstring& label) {
    return label == L"tcp4" || label == L"tcp6" || label == L"udp4" || label == L"udp6";
}

int child_control(const std::wstring& label, USHORT port) {
    require(known_label(label), "Unknown positive control");
    private_machine();
    Winsock winsock;
    const bool ipv6 = label.back() == L'6', tcp = label.substr(0, 3) == L"tcp";
    Socket handle(socket(ipv6 ? AF_INET6 : AF_INET, tcp ? SOCK_STREAM : SOCK_DGRAM,
                         tcp ? IPPROTO_TCP : IPPROTO_UDP));
    sockaddr_storage address{};
    int size;
    destination(ipv6, port, address, size);
    if (tcp) {
        u_long nonblocking = 1;
        require(ioctlsocket(handle.value, FIONBIO, &nonblocking) == 0, "Cannot bound TCP control");
        const int result = connect(handle.value, reinterpret_cast<sockaddr*>(&address), size);
        if (result != 0) {
            require(WSAGetLastError() == WSAEWOULDBLOCK, "TCP control connect failed");
            fd_set writable, failed;
            FD_ZERO(&writable); FD_ZERO(&failed);
            FD_SET(handle.value, &writable); FD_SET(handle.value, &failed);
            timeval timeout{1, 0};
            require(select(0, nullptr, &writable, &failed, &timeout) > 0
                    && FD_ISSET(handle.value, &writable) && !FD_ISSET(handle.value, &failed),
                    "TCP control connect did not complete");
            int error = 0, error_size = sizeof(error);
            require(getsockopt(handle.value, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&error),
                    &error_size) == 0 && error == 0, "TCP control reported an error");
        }
    } else {
        const char marker = 'C';
        require(sendto(handle.value, &marker, 1, 0, reinterpret_cast<sockaddr*>(&address), size) == 1,
                "UDP loopback control send failed");
    }
    return 0;
}

std::string control(const std::wstring& label) {
    const bool ipv6 = label.back() == L'6', tcp = label.substr(0, 3) == L"tcp";
    Socket listener(socket(ipv6 ? AF_INET6 : AF_INET, tcp ? SOCK_STREAM : SOCK_DGRAM,
                           tcp ? IPPROTO_TCP : IPPROTO_UDP));
    sockaddr_storage address{};
    int size;
    destination(ipv6, 0, address, size);
    require(bind(listener.value, reinterpret_cast<sockaddr*>(&address), size) == 0,
            "Cannot bind the guest loopback control destination");
    if (tcp) require(listen(listener.value, 1) == 0, "Cannot listen for the TCP control");
    require(getsockname(listener.value, reinterpret_cast<sockaddr*>(&address), &size) == 0,
            "Cannot identify loopback control port");
    const USHORT port = ntohs(ipv6 ? reinterpret_cast<sockaddr_in6*>(&address)->sin6_port
                                  : reinterpret_cast<sockaddr_in*>(&address)->sin_port);
    require(port != 0, "Control destination has no port");
    Handle job(CreateJobObjectW(nullptr, nullptr));
    require(job.value != nullptr, "Cannot create the owned control job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    require(SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) != 0,
            "Cannot bound owned child lifetime");
    const auto path = executable_path();
    std::wstring command = L"\"" + path + L"\" --control " + label + L" " + std::to_wstring(port);
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const auto begin = now();
    require(CreateProcessW(path.c_str(), command.data(), nullptr, nullptr, FALSE,
            CREATE_SUSPENDED | CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process) != 0,
            "Cannot create the exact owned control child");
    Handle child(process.hProcess), child_thread(process.hThread);
    if (!AssignProcessToJobObject(job.value, child.value)) {
        TerminateProcess(child.value, 3); // Only the child just created suspended.
        require(WaitForSingleObject(child.value, 2000) == WAIT_OBJECT_0, "Owned child did not terminate");
        throw std::runtime_error("Cannot isolate the owned control child job");
    }
    require(ResumeThread(child_thread.value) != static_cast<DWORD>(-1), "Cannot start owned control child");
    const DWORD waited = WaitForSingleObject(child.value, 2000);
    if (waited != WAIT_OBJECT_0) {
        TerminateJobObject(job.value, 3);
        require(WaitForSingleObject(child.value, 2000) == WAIT_OBJECT_0, "Timed-out child did not exit");
        throw std::runtime_error("Positive-control child timed out");
    }
    DWORD code = 3;
    require(GetExitCodeProcess(child.value, &code) != 0 && code == 0,
            "Positive-control socket call did not complete successfully");
    const auto end = now();
    const std::string ascii_label = tcp ? (ipv6 ? "tcp6" : "tcp4") : (ipv6 ? "udp6" : "udp4");
    std::ostringstream out;
    out << "{\"label\":" << quoted(ascii_label)
        << ",\"child_pid\":" << process.dwProcessId << ",\"begin_qpc\":" << begin
        << ",\"end_qpc\":" << end << ",\"call_completed\":true,\"child_exited\":true,"
        "\"remote_address\":" << quoted(ipv6 ? "::1" : "127.0.0.1") << ",\"remote_port\":" << port << '}';
    return out.str();
}

std::string positive_control() {
    private_machine();
    reviewed_schemas();
    Winsock winsock;
    Capture data;
    LARGE_INTEGER frequency;
    require(QueryPerformanceFrequency(&frequency) != 0 && frequency.QuadPart > 0,
            "QPC frequency unavailable");
    data.frequency = frequency.QuadPart;
    data.begin = now();
    Trace trace(data);
    trace.start();
    std::vector<std::string> controls;
    for (const auto& label : {L"tcp4", L"tcp6", L"udp4", L"udp6"}) controls.push_back(control(label));
    Sleep(1100); // Bounded flush allowance, not evidence of successful observation.
    trace.stop();
    data.end = now();
    require(data.end - data.begin <= 30 * data.frequency, "Control exceeded thirty-second capture bound");
    const auto consumed = data.consumer_status.load();
    const bool decoded = data.decoder_ok && consumed == ERROR_SUCCESS;
    std::ostringstream out;
    out << "{\"events\":[";
    for (size_t i = 0; i < data.events.size(); ++i) { if (i) out << ','; out << data.events[i]; }
    out << "],\"controls\":[";
    for (size_t i = 0; i < controls.size(); ++i) { if (i) out << ','; out << controls[i]; }
    out << "],\"capture\":{\"vm_uuid\":" << quoted(private_uuid)
        << ",\"trace_started\":" << (trace.started ? "true" : "false")
        << ",\"trace_stopped\":" << (trace.stopped ? "true" : "false")
        << ",\"decoder_ok\":" << (decoded ? "true" : "false")
        << ",\"events_lost\":" << trace.lost_events << ",\"buffers_lost\":" << trace.lost_buffers
        << ",\"qpc_frequency\":" << data.frequency << ",\"begin_qpc\":" << data.begin
        << ",\"end_qpc\":" << data.end
        << ",\"packet_content_captured\":false,\"event_limit_reached\":" << (data.limit ? "true" : "false")
        << "}}";
    return out.str();
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (argc == 4 && std::wstring(argv[1]) == L"--control")
            return child_control(argv[2], port_argument(argv[3]));
        require(argc == 2 && std::wstring(argv[1]) == L"--positive-control-only",
                "Only the exact private-VM positive control is supported");
        std::cout << positive_control() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"error\":" << quoted(error.what()) << ",\"privacy_verified\":false,"
            "\"vision_runtime_loaded\":false,\"gpu_benchmarked\":false,\"cleanup_required\":"
            << (owned_trace_unstopped ? "true" : "false") << "}\n";
        return 1;
    }
}
