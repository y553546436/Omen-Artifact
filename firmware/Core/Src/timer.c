#include "timer.h"

TIM_HandleTypeDef htim2;

extern void Error_Handler(void);

void MX_TIM2_Init(void) {
    __HAL_RCC_TIM2_CLK_ENABLE();

    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 40000-1;
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 0xFFFFFFFF;
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV4;
    htim2.Init.RepetitionCounter = 0;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    if (HAL_TIM_Base_Init(&htim2) != HAL_OK)
    {
        Error_Handler();
    }
}

void timer_start(void) {
    __HAL_TIM_SET_COUNTER(&htim2, 0);  // Reset the counter
    HAL_TIM_Base_Start(&htim2);        // Start the timer
}

uint32_t timer_stop(void) {
    HAL_TIM_Base_Stop(&htim2);         // Stop the timer
    return __HAL_TIM_GET_COUNTER(&htim2);  // Get the counter value
}
