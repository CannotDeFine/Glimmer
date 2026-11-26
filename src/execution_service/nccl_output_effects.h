#pragma once

#include "nccl_actor.h"

namespace glimmer::app {

// Trusted composition only: bind the actual pinned SDK before supplying this
// producer to an actor. Isolated-cell ANY-WRITE evidence plus that SDK's whole-
// typed-field store/host-stability contract authorizes projection. Status and
// byte differences never supply missing evidence. This does not retire owners.
[[nodiscard]] NcclActorEffectProducer make_pinned_nccl_effect_producer();

}  // namespace glimmer::app
