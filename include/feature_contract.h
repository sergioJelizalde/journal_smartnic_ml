#ifndef FEATURE_CONTRACT_H
#define FEATURE_CONTRACT_H

#include "common.h"
#include "app_config.h"

int feature_contract_validate(enum app_profile_type profile,
                              const struct feature_vector *fv,
                              uint16_t window);

#endif
