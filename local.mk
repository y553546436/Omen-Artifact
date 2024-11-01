# A makefile pipeline to run all experiments for a given dataset.

# dataset not empty
ifeq ($(DATASET),)
$(error DATASET is not set)
endif

model_folder := model_$(DATASET)_$(TRAINER)_$(DTYPE)
train_target := src/$(model_folder)
bin_flag := $(if $(filter binary,$(DTYPE)),-b)

header_target := performance/hdc_eval/include/model.h
test_header_target := performance/hdc_eval/include/testdata.h

start_filename := $(if $(START),s$(START))
freq_filename := $(if $(FREQ),f$(FREQ))
alpha_filename := $(if $(ALPHA),a$(subst .,,$(ALPHA)))


result := $(DATASET)_$(TRAINER)_$(DTYPE)_$(STRATEGY)_$(start_filename)_$(freq_filename)_$(alpha_filename).csv
final_target := output/$(result)

all: $(final_target)

$(final_target):
	mkdir -p output
	make -f local.mk run

$(train_target):
	@echo "Training $(DTYPE) $(TRAINER) model on $(DATASET) dataset..."
	@echo "python trainer.py --trainer $(TRAINER) $(bin_flag) --dataset $(DATASET) --dir $(model_folder) $(TRAINER_ARGS)"
	cd src && python trainer.py --trainer $(TRAINER) $(bin_flag) --dataset $(DATASET) --dir $(model_folder) $(TRAINER_ARGS)
	@echo "Done!"

start_arg := $(if $(START),--start $(START))
freq_arg := $(if $(FREQ),--freq $(FREQ))
alpha_arg := $(if $(ALPHA),--alpha $(ALPHA))
bldc_arg := $(if $(and $(filter LDC,$(TRAINER)), $(filter binary,$(DTYPE))),--BLDC)
strategy_arg := $(if $(STRATEGY),--strategy $(STRATEGY))
header: $(train_target)
	@echo "Generating header file..."
	@echo "python generate_header.py --dataset $(DATASET) --data $(model_folder) $(bin_flag) $(strategy_arg) $(start_arg) $(freq_arg) $(alpha_arg) $(bldc_arg) --header $(header_target)"
	cd src && python generate_header.py --dataset $(DATASET) --data $(model_folder) $(bin_flag) $(strategy_arg) $(start_arg) $(freq_arg) $(alpha_arg) $(bldc_arg) --header ../$(header_target)
	@echo "Done!"

test_header: src/$(DATASET).py
	@echo "Generating test header file..."
	@echo "python $(DATASET).py --header $(test_header_target)"
	cd src && python $(DATASET).py --header ../$(test_header_target)
	@echo "Done!"

build: header test_header
	@echo "Building local c code..."
	make -C performance/hdc_eval clean
	make -C performance/hdc_eval
	@echo "Done!"

clean:
	@echo "Cleaning..."
	rm -rf $(train_target) $(header_target) $(test_header_target)
	@echo "Done!"

run: build
	@echo "Running..."
	./performance/hdc_eval/bin/performance > eval_tmp.log && mv eval_tmp.log $(final_target)
	@echo "Done!"

.PHONY: all build clean run

