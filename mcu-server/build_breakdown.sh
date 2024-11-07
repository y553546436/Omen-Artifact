#!/bin/bash

CURRENT_PATH=$(pwd)
# Define variables
WORKSPACE_PATH="$HOME/STM32CubeIDE/workspace_1.15.1"
PROJECT_PATH="${CURRENT_PATH}/../firmware"
PROJECT_NAME="firmware_breakdown"
ELF_FILE="${PROJECT_PATH}/Release/${PROJECT_NAME}.elf"

STM32CUBEIDE_PATH="/Applications/STM32CubeIDE.app/Contents/MacOS/STM32CubeIDE"
STM32CUBEPROGRAMMER_PATH="/Applications/STMicroelectronics/STM32Cube/STM32CubeProgrammer/STM32CubeProgrammer.app/Contents/MacOs/bin/STM32_Programmer_CLI"

# Build the project using STM32CubeIDE headless build
echo "Building the project..."
$STM32CUBEIDE_PATH --launcher.suppressErrors -nosplash -application org.eclipse.cdt.managedbuilder.core.headlessbuild -data $WORKSPACE_PATH -import $PROJECT_PATH -cleanBuild $PROJECT_NAME

# Check if the build was successful
if [ $? -ne 0 ]; then
    echo "Build failed. Exiting."
    exit 1
fi
echo "Build successful."

# Check for the ELF file
if [ ! -f "$ELF_FILE" ]; then
    echo "Error: ELF file not found. Exiting."
    exit 1
fi

# Program the board using STM32_Programmer_CLI
echo "Flashing the binary to the board..."
$STM32CUBEPROGRAMMER_PATH -c port=SWD -w $ELF_FILE 0x08000000 -v -s

# Check if the flashing was successful
if [ $? -ne 0 ]; then
    echo "Flashing failed. Exiting."
    exit 1
fi
echo "Flashing successful."

echo "Build and flash process completed."
