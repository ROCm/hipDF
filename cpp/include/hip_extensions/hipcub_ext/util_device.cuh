#include "hip/hip_runtime.h"
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * \file
 * Properties of a given CUDA device and the corresponding PTX bundle
 */

#pragma once

#include <atomic>
#include <array>
#include <cassert>

/**
 * \addtogroup UtilMgmt
 * @{
 */

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document


/**
 * \brief Alias temporaries to externally-allocated device storage (or simply return the amount of storage needed).
 */
template <int ALLOCATIONS>
__host__ __device__ __forceinline__
hipError_t AliasTemporaries(
    void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
    size_t& temp_storage_bytes,                 ///< [in,out] Size in bytes of \t d_temp_storage allocation
    void*   (&allocations)[ALLOCATIONS],        ///< [in,out] Pointers to device allocations needed
    size_t  (&allocation_sizes)[ALLOCATIONS])   ///< [in] Sizes in bytes of device allocations needed
{
    const int ALIGN_BYTES   = 256;
    const int ALIGN_MASK    = ~(ALIGN_BYTES - 1);

    // Compute exclusive prefix sum over allocation requests
    size_t allocation_offsets[ALLOCATIONS];
    size_t bytes_needed = 0;
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        size_t allocation_bytes = (allocation_sizes[i] + ALIGN_BYTES - 1) & ALIGN_MASK;
        allocation_offsets[i] = bytes_needed;
        bytes_needed += allocation_bytes;
    }
    bytes_needed += ALIGN_BYTES - 1;

    // Check if the caller is simply requesting the size of the storage allocation
    if (!d_temp_storage)
    {
        temp_storage_bytes = bytes_needed;
        return hipSuccess;
    }

    // Check if enough storage provided
    if (temp_storage_bytes < bytes_needed)
    {
        return HipcubDebug(hipErrorInvalidValue);
    }

    // Alias
    d_temp_storage = (void *) ((size_t(d_temp_storage) + ALIGN_BYTES - 1) & ALIGN_MASK);
    for (int i = 0; i < ALLOCATIONS; ++i)
    {
        allocations[i] = static_cast<char*>(d_temp_storage) + allocation_offsets[i];
    }

    return hipSuccess;
}


/**
 * \brief Empty kernel for querying PTX manifest metadata (e.g., version) for the current device
 */
template <typename T>
__global__ void EmptyKernel(void) { }

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/**
 * \brief Returns the current device or -1 if an error occurred.
 */
HIPCUB_RUNTIME_FUNCTION inline int CurrentDevice()
{
#if defined(HIPCUB_RUNTIME_ENABLED) // Host code or device code with the CUDA runtime.

    int device = -1;
    if (HipcubDebug(hipGetDevice(&device))) return -1;
    return device;

#else // Device code without the CUDA runtime.

    return -1;

#endif
}

/**
 * \brief RAII helper which saves the current device and switches to the
 *        specified device on construction and switches to the saved device on
 *        destruction.
 */
struct SwitchDevice
{
private:
    int const old_device;
    bool const needs_reset;
public:
    __host__ inline SwitchDevice(int new_device)
      : old_device(CurrentDevice()), needs_reset(old_device != new_device)
    {
        if (needs_reset)
            auto a = hipSetDevice(new_device);
            // HipcubDebug();
    }

    __host__ inline ~SwitchDevice()
    {
        if (needs_reset)
            auto a = hipSetDevice(old_device);
            // HipcubDebug();
    }
};

/**
 * \brief Returns the number of CUDA devices available or -1 if an error
 *        occurred.
 */
HIPCUB_RUNTIME_FUNCTION inline int DeviceCountUncached()
{
#if defined(HIPCUB_RUNTIME_ENABLED) // Host code or device code with the CUDA runtime.

    int count = -1;
    if (HipcubDebug(hipGetDeviceCount(&count)))
        // CUDA makes no guarantees about the state of the output parameter if
        // `hipGetDeviceCount` fails; in practice, they don't, but out of
        // paranoia we'll reset `count` to `-1`.
        count = -1;
    return count;

#else // Device code without the CUDA runtime.

    return -1;

#endif
}

#if HIPCUB_CPP_DIALECT >= 2011 // C++11 and later.

/**
 * \brief Cache for an arbitrary value produced by a nullary function.
 */
template <typename T, T(*Function)()>
struct ValueCache
{
    T const value;

    /**
     * \brief Call the nullary function to produce the value and construct the
     *        cache.
     */
    __host__ inline ValueCache() : value(Function()) {}
};

#endif

#if HIPCUB_CPP_DIALECT >= 2011
// Host code, only safely usable in C++11 or newer, where thread-safe
// initialization of static locals is guaranteed.  This is a separate function
// to avoid defining a local static in a host/device function.
__host__ inline int DeviceCountCachedValue()
{
    static ValueCache<int, DeviceCountUncached> cache;
    return cache.value;
}
#endif

/**
 * \brief Returns the number of CUDA devices available.
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
HIPCUB_RUNTIME_FUNCTION inline int DeviceCount()
{
    int result = -1;
    if (HIPCUB_IS_HOST_CODE) {
        #if HIPCUB_INCLUDE_HOST_CODE
            #if HIPCUB_CPP_DIALECT >= 2011
                // Host code and C++11.
                result = DeviceCountCachedValue();
            #else
                // Host code and C++98.
                result = DeviceCountUncached();
            #endif
        #endif
    } else {
        #if HIPCUB_INCLUDE_DEVICE_CODE
            // Device code.
            result = DeviceCountUncached();
        #endif
    }
    return result;
}

#if HIPCUB_CPP_DIALECT >= 2011 // C++11 and later.

/**
 * \brief Per-device cache for a CUDA attribute value; the attribute is queried
 *        and stored for each device upon construction.
 */
struct PerDeviceAttributeCache
{
    struct DevicePayload
    {
        int         attribute;
        hipError_t error;
    };

    // Each entry starts in the `DeviceEntryEmpty` state, then proceeds to the
    // `DeviceEntryInitializing` state, and then proceeds to the
    // `DeviceEntryReady` state. These are the only state transitions allowed;
    // e.g. a linear sequence of transitions.
    enum DeviceEntryStatus
    {
        DeviceEntryEmpty = 0,
        DeviceEntryInitializing,
        DeviceEntryReady
    };

    struct DeviceEntry
    {
        std::atomic<DeviceEntryStatus> flag;
        DevicePayload                  payload;
    };

private:
    std::array<DeviceEntry, HIPCUB_MAX_DEVICES> entries_;

public:
    /**
     * \brief Construct the cache.
     */
    __host__ inline PerDeviceAttributeCache() : entries_()
    {
        assert(DeviceCount() <= HIPCUB_MAX_DEVICES);
    }

    /**
     * \brief Retrieves the payload of the cached function \p f for \p device.
     *
     * \note You must pass a morally equivalent function in to every call or
     *       this function has undefined behavior.
     */
    template <typename Invocable>
    __host__ DevicePayload operator()(Invocable&& f, int device)
    {
        if (device >= DeviceCount())
            return DevicePayload{0, hipErrorInvalidDevice};

        auto& entry   = entries_[device];
        auto& flag    = entry.flag;
        auto& payload = entry.payload;

        DeviceEntryStatus old_status = DeviceEntryEmpty;

        // First, check for the common case of the entry being ready.
        if (flag.load(std::memory_order_acquire) != DeviceEntryReady)
        {
            // Assume the entry is empty and attempt to lock it so we can fill
            // it by trying to set the state from `DeviceEntryReady` to
            // `DeviceEntryInitializing`.
            if (flag.compare_exchange_strong(old_status, DeviceEntryInitializing,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire))
            {
                // We successfully set the state to `DeviceEntryInitializing`;
                // we have the lock and it's our job to initialize this entry
                // and then release it.

                // We don't use `HipcubDebug` here because we let the user code
                // decide whether or not errors are hard errors.
                payload.error = std::forward<Invocable>(f)(payload.attribute);
                if (payload.error)
                    // Clear the global CUDA error state which may have been
                    // set by the last call. Otherwise, errors may "leak" to
                    // unrelated kernel launches.
                    hipGetLastError();

                // Release the lock by setting the state to `DeviceEntryReady`.
                flag.store(DeviceEntryReady, std::memory_order_release);
            }

            // If the `compare_exchange_weak` failed, then `old_status` has
            // been updated with the value of `flag` that it observed.

            else if (old_status == DeviceEntryInitializing)
            {
                // Another execution agent is initializing this entry; we need
                // to wait for them to finish; we'll know they're done when we
                // observe the entry status as `DeviceEntryReady`.
                do { old_status = flag.load(std::memory_order_acquire); }
                while (old_status != DeviceEntryReady);
                // FIXME: Use `atomic::wait` instead when we have access to
                // host-side C++20 atomics. We could use libcu++, but it only
                // supports atomics for SM60 and up, even if you're only using
                // them in host code.
            }
        }

        // We now know that the state of our entry is `DeviceEntryReady`, so
        // just return the entry's payload.
        return entry.payload;
    }
};

#endif

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 */
HIPCUB_RUNTIME_FUNCTION inline hipError_t PtxVersionUncached(int& ptx_version)
{
    // Instantiate `EmptyKernel<void>` in both host and device code to ensure
    // it can be called.
    typedef void (*EmptyKernelPtr)();
    EmptyKernelPtr empty_kernel = EmptyKernel<void>;

    // This is necessary for unused variable warnings in host compilers. The
    // usual syntax of (void)empty_kernel; was not sufficient on MSVC2015.
    (void)reinterpret_cast<void*>(empty_kernel);

    hipError_t result = hipSuccess;
    if (HIPCUB_IS_HOST_CODE) {
       #if HIPCUB_INCLUDE_HOST_CODE
            hipFuncAttributes empty_kernel_attrs;

            result = hipFuncGetAttributes(&empty_kernel_attrs,
                                           reinterpret_cast<void*>(empty_kernel));
            CUDF_CUDA_TRY(HipcubDebug(result));

            ptx_version = empty_kernel_attrs.ptxVersion * 10;
        #endif
    } else {
        #if HIPCUB_INCLUDE_DEVICE_CODE
            // This is necessary to ensure instantiation of EmptyKernel in device code.
            // The `reinterpret_cast` is necessary to suppress a set-but-unused warnings.
            // This is a meme now: https://twitter.com/blelbach/status/1222391615576100864
            (void)reinterpret_cast<EmptyKernelPtr>(empty_kernel);

            ptx_version = HIPCUB_ARCH;
        #endif
    }
    return result;
}

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 */
__host__ inline hipError_t PtxVersionUncached(int& ptx_version, int device)
{
    SwitchDevice sd(device);
    (void)sd;
    return PtxVersionUncached(ptx_version);
}

#if HIPCUB_CPP_DIALECT >= 2011 // C++11 and later.
template <typename Tag>
__host__ inline PerDeviceAttributeCache& GetPerDeviceAttributeCache()
{
    // C++11 guarantees that initialization of static locals is thread safe.
    static PerDeviceAttributeCache cache;
    return cache;
}

struct PtxVersionCacheTag {};
struct SmVersionCacheTag {};
#endif

/**
 * \brief Retrieves the PTX version that will be used on \p device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
__host__ inline hipError_t PtxVersion(int& ptx_version, int device)
{
#if HIPCUB_CPP_DIALECT >= 2011 // C++11 and later.

    auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
      // If this call fails, then we get the error code back in the payload,
      // which we check with `HipcubDebug` below.
      [=] (int& pv) { return PtxVersionUncached(pv, device); },
      device);

    if (!HipcubDebug(payload.error))
        ptx_version = payload.attribute;

    return payload.error;

#else // Pre C++11.

    return PtxVersionUncached(ptx_version, device);

#endif
}

/**
 * \brief Retrieves the PTX version that will be used on the current device (major * 100 + minor * 10).
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
HIPCUB_RUNTIME_FUNCTION inline hipError_t PtxVersion(int& ptx_version)
{
    hipError_t result = hipErrorUnknown;
    if (HIPCUB_IS_HOST_CODE) {
        #if HIPCUB_INCLUDE_HOST_CODE
            #if HIPCUB_CPP_DIALECT >= 2011
                // Host code and C++11.
                auto const device = CurrentDevice();

                auto const payload = GetPerDeviceAttributeCache<PtxVersionCacheTag>()(
                  // If this call fails, then we get the error code back in the payload,
                  // which we check with `HipcubDebug` below.
                  [=] (int& pv) { return PtxVersionUncached(pv, device); },
                  device);

                if (!HipcubDebug(payload.error))
                    ptx_version = payload.attribute;

                result = payload.error;
            #else
                // Host code and C++98.
                result = PtxVersionUncached(ptx_version);
            #endif
        #endif
    } else {
        #if HIPCUB_INCLUDE_DEVICE_CODE
            // Device code.
            result = PtxVersionUncached(ptx_version);
        #endif
    }
    return result;
}

/**
 * \brief Retrieves the SM version of \p device (major * 100 + minor * 10)
 */
HIPCUB_RUNTIME_FUNCTION inline hipError_t SmVersionUncached(int& sm_version, int device = CurrentDevice())
{
#if defined(HIPCUB_RUNTIME_ENABLED) // Host code or device code with the CUDA runtime.

    hipError_t error = hipSuccess;
    do
    {
        int major = 0, minor = 0;
        if (HipcubDebug(error = hipDeviceGetAttribute(&major, hipDeviceAttributeComputeCapabilityMajor, device))) break;
        if (HipcubDebug(error = hipDeviceGetAttribute(&minor, hipDeviceAttributeComputeCapabilityMinor, device))) break;
        sm_version = major * 100 + minor * 10;
    }
    while (0);

    return error;

#else // Device code without the CUDA runtime.

    (void)sm_version;
    (void)device;

    // CUDA API calls are not supported from this device.
    return HipcubDebug(hipErrorInvalidConfiguration);

#endif
}

/**
 * \brief Retrieves the SM version of \p device (major * 100 + minor * 10)
 *
 * \note This function may cache the result internally.
 *
 * \note This function is thread safe.
 */
HIPCUB_RUNTIME_FUNCTION inline hipError_t SmVersion(int& sm_version, int device = CurrentDevice())
{
    hipError_t result = hipErrorUnknown;
    if (HIPCUB_IS_HOST_CODE) {
        #if HIPCUB_INCLUDE_HOST_CODE
            #if HIPCUB_CPP_DIALECT >= 2011
                // Host code and C++11
                auto const payload = GetPerDeviceAttributeCache<SmVersionCacheTag>()(
                  // If this call fails, then we get the error code back in the payload,
                  // which we check with `HipcubDebug` below.
                  [=] (int& pv) { return SmVersionUncached(pv, device); },
                  device);

                if (!HipcubDebug(payload.error))
                    sm_version = payload.attribute;

                result = payload.error;
            #else
                // Host code and C++98
                result = SmVersionUncached(sm_version, device);
            #endif
        #endif
    } else {
        #if HIPCUB_INCLUDE_DEVICE_CODE
            result = SmVersionUncached(sm_version, device);
        #endif
    }
    return result;
}

/**
 * Synchronize the specified \p stream.
 */
HIPCUB_RUNTIME_FUNCTION inline hipError_t SyncStream(hipStream_t stream)
{
    hipError_t result = hipErrorUnknown;
    if (HIPCUB_IS_HOST_CODE) {
        #if HIPCUB_INCLUDE_HOST_CODE
            result = HipcubDebug(hipStreamSynchronize(stream));
        #endif
    } else {
        #if HIPCUB_INCLUDE_DEVICE_CODE
            #if defined(HIPCUB_RUNTIME_ENABLED) // Device code with the CUDA runtime.
                (void)stream;
                // Device can't yet sync on a specific stream
                result = HipcubDebug(hipcub_extensions::detail::device_synchronize());
            #else // Device code without the CUDA runtime.
                (void)stream;
                // CUDA API calls are not supported from this device.
                result = HipcubDebug(hipErrorInvalidConfiguration);
            #endif
        #endif
    }
    return result;
}


/**
 * \brief Computes maximum SM occupancy in thread blocks for executing the given kernel function pointer \p kernel_ptr on the current device with \p block_threads per thread block.
 *
 * \par Snippet
 * The code snippet below illustrates the use of the MaxSmOccupancy function.
 * \par
 * \code
 * #include <hipcub/hipcub.hpp>   // or equivalently <cub/util_device.cuh>
 *
 * template <typename T>
 * __global__ void ExampleKernel()
 * {
 *     // Allocate shared memory for BlockScan
 *     __shared__ volatile T buffer[4096];
 *
 *        ...
 * }
 *
 *     ...
 *
 * // Determine SM occupancy for ExampleKernel specialized for unsigned char
 * int max_sm_occupancy;
 * MaxSmOccupancy(max_sm_occupancy, ExampleKernel<unsigned char>, 64);
 *
 * // max_sm_occupancy  <-- 4 on SM10
 * // max_sm_occupancy  <-- 8 on SM20
 * // max_sm_occupancy  <-- 12 on SM35
 *
 * \endcode
 *
 */
template <typename KernelPtr>
HIPCUB_RUNTIME_FUNCTION inline
hipError_t MaxSmOccupancy(
    int&                max_sm_occupancy,          ///< [out] maximum number of thread blocks that can reside on a single SM
    KernelPtr           kernel_ptr,                 ///< [in] Kernel pointer for which to compute SM occupancy
    int                 block_threads,              ///< [in] Number of threads per thread block
    int                 dynamic_smem_bytes = 0)
{
#ifndef HIPCUB_RUNTIME_ENABLED

    (void)dynamic_smem_bytes;
    (void)block_threads;
    (void)kernel_ptr;
    (void)max_sm_occupancy;

    // CUDA API calls not supported from this device
    return HipcubDebug(hipErrorInvalidConfiguration);

#else

    return HipcubDebug(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_sm_occupancy,
        kernel_ptr,
        block_threads,
        dynamic_smem_bytes));

#endif  // HIPCUB_RUNTIME_ENABLED
}


/******************************************************************************
 * Policy management
 ******************************************************************************/

/**
 * Kernel dispatch configuration
 */
struct KernelConfig
{
    int block_threads;
    int items_per_thread;
    int tile_size;
    int sm_occupancy;

    HIPCUB_RUNTIME_FUNCTION __forceinline__
    KernelConfig() : block_threads(0), items_per_thread(0), tile_size(0), sm_occupancy(0) {}

    template <typename AgentPolicyT, typename KernelPtrT>
    HIPCUB_RUNTIME_FUNCTION __forceinline__
    hipError_t Init(KernelPtrT kernel_ptr)
    {
        block_threads        = AgentPolicyT::BLOCK_THREADS;
        items_per_thread     = AgentPolicyT::ITEMS_PER_THREAD;
        tile_size            = block_threads * items_per_thread;
        hipError_t retval   = MaxSmOccupancy(sm_occupancy, kernel_ptr, block_threads);
        return retval;
    }
};



/// Helper for dispatching into a policy chain
template <int PTX_VERSION, typename PolicyT, typename PrevPolicyT>
struct ChainedPolicy
{
  /// The policy for the active compiler pass
  using ActivePolicy = 
    hipcub_extensions::detail::conditional_t<(HIPCUB_ARCH < PTX_VERSION),
                               typename PrevPolicyT::ActivePolicy,
                               PolicyT>;

  /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
  template <typename FunctorT>
  HIPCUB_RUNTIME_FUNCTION __forceinline__
  static hipError_t Invoke(int ptx_version, FunctorT& op)
  {
      if (ptx_version < PTX_VERSION) {
          return PrevPolicyT::Invoke(ptx_version, op);
      }
      return op.template Invoke<PolicyT>();
  }
};

/// Helper for dispatching into a policy chain (end-of-chain specialization)
template <int PTX_VERSION, typename PolicyT>
struct ChainedPolicy<PTX_VERSION, PolicyT, PolicyT>
{
    /// The policy for the active compiler pass
    typedef PolicyT ActivePolicy;

    /// Specializes and dispatches op in accordance to the first policy in the chain of adequate PTX version
    template <typename FunctorT>
    HIPCUB_RUNTIME_FUNCTION __forceinline__
    static hipError_t Invoke(int /*ptx_version*/, FunctorT& op) {
        return op.template Invoke<PolicyT>();
    }
};




/** @} */       // end group UtilMgmt
