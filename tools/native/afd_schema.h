// Shared bounded metadata reader. Merely including it does not enable tracing.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <evntcons.h>
#include <tdh.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
GUID provider = {0xe53c6823, 0x7bb8, 0x44bb, {0x90, 0xdc, 0x3f, 0x86, 0x09, 0x0d, 0x48, 0xa6}};
constexpr ULONG max_metadata = 1024 * 1024;
constexpr std::array<USHORT, 7> selected = {1, 4, 5, 13, 14, 21, 22};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(ULONG status, const char* operation) {
    if (status != ERROR_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
}

std::string quoted(const std::string& value) {
    std::ostringstream out;
    out << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << c;
    }
    out << '"';
    return out.str();
}

std::string text_at(const std::vector<BYTE>& buffer, ULONG offset) {
    require(offset >= offsetof(TRACE_EVENT_INFO, EventPropertyInfoArray)
            && offset < buffer.size() && offset % sizeof(wchar_t) == 0,
            "Invalid metadata text offset");
    std::wstring text;
    bool terminated = false;
    for (size_t i = offset; i + sizeof(wchar_t) <= buffer.size(); i += sizeof(wchar_t)) {
        wchar_t unit;
        std::memcpy(&unit, buffer.data() + i, sizeof(unit));
        if (unit == 0) { terminated = true; break; }
        text.push_back(unit);
        require(text.size() <= 4096, "Metadata text exceeds bound");
    }
    require(terminated, "Unterminated metadata text");
    if (text.empty()) return {};
    const int count = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
    require(count > 0, "Invalid metadata UTF-16");
    std::string result(static_cast<size_t>(count), '\0');
    require(WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
        static_cast<int>(text.size()), result.data(), count, nullptr, nullptr) == count,
        "Metadata UTF-8 conversion failed");
    return result;
}

std::vector<EVENT_DESCRIPTOR> descriptors() {
    ULONG size = 0;
    require(TdhEnumerateManifestProviderEvents(&provider, nullptr, &size)
            == ERROR_INSUFFICIENT_BUFFER && size >= sizeof(PROVIDER_EVENT_INFO)
            && size <= max_metadata, "Cannot size AFD manifest event inventory");
    std::vector<BYTE> buffer(size);
    check(TdhEnumerateManifestProviderEvents(&provider,
        reinterpret_cast<PROVIDER_EVENT_INFO*>(buffer.data()), &size), "Enumerate AFD events");
    require(size <= buffer.size(), "Metadata inventory grew unexpectedly");
    buffer.resize(size);
    const auto* info = reinterpret_cast<const PROVIDER_EVENT_INFO*>(buffer.data());
    const size_t offset = offsetof(PROVIDER_EVENT_INFO, EventDescriptorsArray);
    require(info->Reserved == 0 && info->NumberOfEvents <= 4096
            && offset + static_cast<size_t>(info->NumberOfEvents) * sizeof(EVENT_DESCRIPTOR)
                <= buffer.size(), "Invalid manifest inventory bounds");
    return {info->EventDescriptorsArray, info->EventDescriptorsArray + info->NumberOfEvents};
}

std::string schema(EVENT_DESCRIPTOR descriptor) {
    ULONG size = 0;
    require(TdhGetManifestEventInformation(&provider, &descriptor, nullptr, &size)
            == ERROR_INSUFFICIENT_BUFFER && size >= sizeof(TRACE_EVENT_INFO)
            && size <= max_metadata, "Cannot size AFD event schema");
    std::vector<BYTE> buffer(size);
    check(TdhGetManifestEventInformation(&provider, &descriptor,
        reinterpret_cast<TRACE_EVENT_INFO*>(buffer.data()), &size), "Read AFD event schema");
    require(size <= buffer.size(), "Event schema grew unexpectedly");
    buffer.resize(size);
    const auto* info = reinterpret_cast<const TRACE_EVENT_INFO*>(buffer.data());
    require(IsEqualGUID(info->ProviderGuid, provider)
            && std::memcmp(&info->EventDescriptor, &descriptor, sizeof(descriptor)) == 0,
            "Event schema identity changed");
    require(info->DecodingSource == DecodingSourceXMLFile && info->PropertyCount <= 256
            && info->TopLevelPropertyCount <= info->PropertyCount
            && offsetof(TRACE_EVENT_INFO, EventPropertyInfoArray)
                + static_cast<size_t>(info->PropertyCount) * sizeof(EVENT_PROPERTY_INFO)
                    <= buffer.size(), "Unsupported or truncated manifest schema");
    std::ostringstream out;
    out << "{\"event_id\":" << descriptor.Id
        << ",\"version\":" << static_cast<unsigned>(descriptor.Version)
        << ",\"channel\":" << static_cast<unsigned>(descriptor.Channel)
        << ",\"level\":" << static_cast<unsigned>(descriptor.Level)
        << ",\"opcode\":" << static_cast<unsigned>(descriptor.Opcode)
        << ",\"task\":" << descriptor.Task
        << ",\"keywords\":" << descriptor.Keyword
        << ",\"top_level_properties\":" << info->TopLevelPropertyCount
        << ",\"properties\":[";
    for (ULONG i = 0; i < info->PropertyCount; ++i) {
        const auto& property = info->EventPropertyInfoArray[i];
        // Structured fields need separate review; never reinterpret the union.
        require((property.Flags & PropertyStruct) == 0, "Structured event field needs review");
        if (i) out << ',';
        out << "{\"name\":" << quoted(text_at(buffer, property.NameOffset))
            << ",\"flags\":" << static_cast<ULONG>(property.Flags)
            << ",\"in_type\":" << property.nonStructType.InType
            << ",\"out_type\":" << property.nonStructType.OutType
            << ",\"count_or_index\":" << property.count
            << ",\"length_or_index\":" << property.length << '}';
    }
    out << "]}";
    return out.str();
}
}  // namespace

#ifdef CARL_CAM_SCHEMA_MAIN
int wmain(int argc, wchar_t** argv) {
    if (argc != 2 || std::wstring(argv[1]) != L"--schema-only") {
        std::cerr << "This read-only utility requires --schema-only. It cannot start a trace.\n";
        return 2;
    }
    try {
        const auto all = descriptors();
        std::ostringstream result;
        result << "{\"schema_version\":1,\"provider\":\"e53c6823-7bb8-44bb-90dc-3f86090d48a6\","
            "\"metadata_scope\":\"executing_machine\",\"trace_started\":false,"
            "\"packet_content_captured\":false,\"vision_runtime_loaded\":false,"
            "\"privacy_verified\":false,\"gpu_benchmarked\":false,\"events\":[";
        for (size_t index = 0; index < selected.size(); ++index) {
            std::vector<EVENT_DESCRIPTOR> matches;
            for (const auto& descriptor : all)
                if (descriptor.Id == selected[index] && descriptor.Version == 0)
                    matches.push_back(descriptor);
            require(matches.size() == 1, "Required version-zero event missing or ambiguous");
            if (index) result << ',';
            result << schema(matches.front());
        }
        result << "]}\n";
        // Emit one complete document only after every selected schema succeeds.
        std::cout << result.str();
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema_version\":1,\"trace_started\":false,"
            "\"privacy_verified\":false,\"error\":" << quoted(error.what()) << "}\n";
        return 1;
    }
}
#endif
