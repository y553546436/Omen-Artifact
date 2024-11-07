#ifndef TIMER_H
#define TIMER_H

#include "stm32u5xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

extern TIM_HandleTypeDef htim2;

void MX_TIM2_Init(void);
void timer_start(void);
uint32_t timer_stop(void);

#ifdef __cplusplus
}
#endif

#endif // TIMER_H

