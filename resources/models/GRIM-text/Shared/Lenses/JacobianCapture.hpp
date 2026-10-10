#pragma once
#include "LensMetadata.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace GRIM::Lenses {
inline std::uint64_t hiddenJacobianBytes(int width,int positions,int layers) {
    if(width<=0 || positions<=0 || layers<=0)throw std::invalid_argument("Invalid Jacobian geometry");
    std::uint64_t bytes=sizeof(float);
    for(int factor:{width,width,positions,layers}) {
        if(bytes>std::numeric_limits<std::uint64_t>::max()/factor)throw std::invalid_argument("Jacobian size overflow");
        bytes*=factor;
    }
    return bytes;
}
inline void appendHiddenJacobianRow(LensCaptureResult& destination,const LensCaptureResult& source,int row) {
    if(source.snapshots.empty() || !source.jacobian_target_position || source.jacobian_rows!=1)
        throw std::invalid_argument("Missing seeded Jacobian capture");
    const int width=source.snapshots.front().metadata.d_model;
    if(row<0 || row>=width || row!=destination.jacobian_rows)
        throw std::invalid_argument("Jacobian rows must be assembled in order");
    if(destination.snapshots.empty()) {
        destination=source;destination.jacobian_rows=0;destination.jacobian_output_dimension.reset();
        for(auto& snapshot:destination.snapshots)snapshot.hidden_jacobian.assign(size_t(width)*width,0.f);
    }
    if(destination.snapshots.size()!=source.snapshots.size() || destination.jacobian_target_position!=source.jacobian_target_position)
        throw std::invalid_argument("Jacobian capture coverage changed between rows");
    for(size_t i=0;i<source.snapshots.size();++i) {
        const auto& input=source.snapshots[i];auto& output=destination.snapshots[i];
        if(input.metadata.d_model!=width || output.metadata.d_model!=width ||
           input.metadata.layer_index!=output.metadata.layer_index || input.metadata.token_position!=output.metadata.token_position ||
           input.metadata.identity.checkpoint_fingerprint!=output.metadata.identity.checkpoint_fingerprint ||
           input.metadata.identity.compiled_config_fingerprint!=output.metadata.identity.compiled_config_fingerprint ||
           input.metadata.identity.parameter_revision!=output.metadata.identity.parameter_revision ||
           input.hidden_jacobian.size()!=size_t(width) || output.hidden_jacobian.size()!=size_t(width)*width)
            throw std::invalid_argument("Jacobian row shape or coordinates changed");
        for(float value:input.hidden_jacobian)if(!std::isfinite(value))throw std::invalid_argument("Nonfinite Jacobian derivative");
        std::copy(input.hidden_jacobian.begin(),input.hidden_jacobian.end(),output.hidden_jacobian.begin()+size_t(row)*width);
    }
    ++destination.jacobian_rows;
}
// Exact IEEE-754 FP32 bit patterns, eight hex digits per entry (most significant
// digit first). This avoids lossy JSON rounding and large decimal arrays.
inline std::string encodeJacobian(const std::vector<float>& values) {
    static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
    constexpr char digits[]="0123456789abcdef";
    std::string output(values.size()*8,'0');
    for(size_t i=0;i<values.size();++i) {
        if(!std::isfinite(values[i]))throw std::invalid_argument("Nonfinite Jacobian derivative");
        std::uint32_t bits;std::memcpy(&bits,&values[i],4);
        for(int digit=0;digit<8;++digit)output[i*8+digit]=digits[(bits>>(28-4*digit))&15];
    }
    return output;
}
inline std::vector<float> decodeJacobian(const std::string& text,size_t entries) {
    if(entries>std::numeric_limits<size_t>::max()/8 || text.size()!=entries*8)
        throw std::invalid_argument("Jacobian matrix shape/encoding mismatch");
    std::vector<float> result(entries);
    for(size_t i=0;i<entries;++i) {
        std::uint32_t bits=0;
        for(size_t digit=0;digit<8;++digit) {
            const char c=text[i*8+digit];
            const int value=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:-1;
            if(value<0)throw std::invalid_argument("Invalid Jacobian hexadecimal encoding");
            bits=(bits<<4)|unsigned(value);
        }
        std::memcpy(&result[i],&bits,4);
        if(!std::isfinite(result[i]))throw std::invalid_argument("Nonfinite Jacobian derivative");
    }
    return result;
}
}
