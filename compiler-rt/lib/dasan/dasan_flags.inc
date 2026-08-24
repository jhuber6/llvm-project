//===-- dasan_flags.inc -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Device address sanitizer runtime flags.
//
//===----------------------------------------------------------------------===//

#ifndef DASAN_FLAG
#  error "Define DASAN_FLAG prior to including this file!"
#endif

DASAN_FLAG(bool, alias_globals, true,
           "Map each loaded code object a second time inside the space so its "
           "globals are checked. Off leaves them at the addresses the loader "
           "chose, where the check cannot see them.")

DASAN_FLAG(int, quarantine_size_mb, 256,
           "How much freed device memory to hold back, across all devices, "
           "before its chunks may be handed out again. A use after free is "
           "only reported while the chunk it names is still in here.")

DASAN_FLAG(int, halt_on_error, -1,
           "Stop the program at the first report. The default of -1 follows "
           "the build: an access compiled with -fsanitize-recover=daddress "
           "keeps going, one compiled without it cannot and so stops. Setting "
           "0 against a build that cannot recover only means the device traps "
           "without the host having stopped first.")

DASAN_FLAG(bool, print_stats, false,
           "Print how many allocations were placed, and why the rest were not, "
           "when the device runtime shuts down.")
