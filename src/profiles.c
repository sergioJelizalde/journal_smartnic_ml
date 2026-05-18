#include "profiles.h"

const struct profile_ops *profile_get(enum app_profile_type type)
{
    switch (type) {
    case APP_PROFILE_IOT: return &iot_profile_ops;
    case APP_PROFILE_DOH: return &doh_profile_ops;
    case APP_PROFILE_ICS: return &ics_profile_ops;
    default: return NULL;
    }
}
