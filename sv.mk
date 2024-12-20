# A makefile pipeline to run all experiments for a given dataset.

# dataset not empty
ifeq ($(DATASET),)
$(error DATASET is not set)
endif

model_folder := model_$(DATASET)_$(TRAINER)_$(DTYPE)
train_target := src/$(model_folder)
bin_flag := $(if $(filter binary,$(DTYPE)),-b)

header_target := firmware/Core/Inc/model.h

testset_folder := mcu-server/data/$(DATASET)
testset_target := $(testset_folder)/test.npy $(testset_folder)/testlbl.npy

start_filename := $(if $(START),s$(START))
freq_filename := $(if $(FREQ),f$(FREQ))
alpha_filename := $(if $(ALPHA),a$(subst .,,$(ALPHA)))


mcu_result := sv_$(DATASET)_$(TRAINER)_$(DTYPE)_$(STRATEGY)_$(start_filename)_$(freq_filename)_$(alpha_filename).csv
final_target := mcu-output_sv/$(mcu_result)

all: $(final_target)

$(final_target):
	@mkdir -p mcu-output_sv
	@make -f sv.mk run
	@mv mcu-server/$(mcu_result) $(final_target)

$(train_target):
	@echo "Training $(DTYPE) $(TRAINER) model on $(DATASET) dataset..."
	@echo "python trainer.py --trainer $(TRAINER) $(bin_flag) --dataset $(DATASET) --dir $(model_folder) $(TRAINER_ARGS)"
	@cd src && python trainer.py --trainer $(TRAINER) $(bin_flag) --dataset $(DATASET) --dir $(model_folder) $(TRAINER_ARGS)
	@echo "Done!"

start_arg := $(if $(START),--start $(START))
freq_arg := $(if $(FREQ),--freq $(FREQ))
alpha_arg := $(if $(ALPHA),--alpha $(ALPHA))
bldc_arg := $(if $(and $(filter LDC,$(TRAINER)), $(filter binary,$(DTYPE))),--BLDC)
strategy_arg := $(if $(STRATEGY),--strategy $(STRATEGY))
header: $(train_target)
	@echo "Generating header file..."
	@echo "python generate_header.py --dataset $(DATASET) --data $(model_folder) $(bin_flag) $(strategy_arg) $(start_arg) $(freq_arg) $(alpha_arg) $(bldc_arg) --header $(header_target)"
	@cd src && python generate_header.py --dataset $(DATASET) --data $(model_folder) $(bin_flag) $(strategy_arg) $(start_arg) $(freq_arg) $(alpha_arg) $(bldc_arg) --header ../$(header_target) --cutoff $(CUTOFF)
	@echo "Done!"

$(testset_target): src/$(DATASET).py
	@echo "Generating test set..."
	@echo "python $(DATASET).py --numpy $(testset_folder)"
	@cd src && python $(DATASET).py --numpy ../$(testset_folder)
	@echo "Done!"

build: header
	@echo "Building firmware..."
	@cd mcu-server && ./build.sh
	@echo "Done!"

clean:
	@echo "Cleaning..."
	@rm -rf $(train_target) $(header_target) $(testset_target) firmware/Debug firmware/Release $(final_target)
	@echo "Done!"

run: build $(testset_target)
	@echo "Running..."
	@cd mcu-server && python3 server.py --dataset $(DATASET) --device $(DEVICE) --output $(mcu_result)
	@echo "Done!"

.PHONY: all build clean run

