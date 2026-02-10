#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "testy/test.h"
#include "ann_hypertune.h"

#if defined(USE_CBLAS)
#	include <cblas.h>
#endif

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

#if defined(USE_CBLAS)
	cblas_init(CBLAS_DEFAULT_THREADS);
	printf( "%s\n", cblas_get_config());
	printf("      CPU uArch: %s\n", cblas_get_corename());
	printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
#endif

    MODULE("Hypertuning Tests");

    // ========================================================================
    SUITE("Initialization");
    COMMENT("Testing initialization functions...");

    HyperparamSpace space;
    hypertune_space_init(&space);
    TESTEX("space_init sets learning_rate_min", (space.learning_rate_min > 0));
    TESTEX("space_init sets learning_rate_max", (space.learning_rate_max > space.learning_rate_min));
    TESTEX("space_init sets batch_size_count", (space.batch_size_count > 0));
    TESTEX("space_init sets optimizer_count", (space.optimizer_count > 0));
    TESTEX("space_init sets hidden_layer_count_options", (space.hidden_layer_count_options > 0));
    TESTEX("space_init sets epoch_limit", (space.epoch_limit > 0));

    HypertuneOptions options;
    hypertune_options_init(&options);
    TESTEX("options_init sets score_func", (options.score_func != NULL));
    TESTEX("options_init sets verbosity to 1", (options.verbosity == 1));

    HypertuneResult result;
    hypertune_result_init(&result);
    TESTEX("result_init sets score to -1", (result.score < 0));
    TESTEX("result_init sets trial_id to 0", (result.trial_id == 0));

    // ========================================================================
    SUITE("Grid Trial Counting");
    COMMENT("Testing trial count calculation...");

    // Reset to known values
    hypertune_space_init(&space);
    space.learning_rate_steps = 2;
    space.batch_size_count = 2;
    space.optimizer_count = 2;
    space.hidden_layer_count_options = 2;
    space.hidden_layer_size_count = 2;
    space.hidden_activation_count = 2;

    int expected = 2 * 2 * 2 * 2 * 2 * 2;  // 64 combinations
    int count = hypertune_count_grid_trials(&space);
    TESTEX("Grid trial count matches expected", (count == expected));

    // Single value for each
    space.learning_rate_steps = 1;
    space.batch_size_count = 1;
    space.optimizer_count = 1;
    space.hidden_layer_count_options = 1;
    space.hidden_layer_size_count = 1;
    space.hidden_activation_count = 1;
    count = hypertune_count_grid_trials(&space);
    TESTEX("Single combination returns 1", (count == 1));

    // ========================================================================
    SUITE("Data Splitting");
    COMMENT("Testing train/validation split...");

    // Create test data
    PTensor inputs = tensor_create(100, 4);
    PTensor outputs = tensor_create(100, 2);
    
    // Fill with sequential data for verification
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 4; j++)
            inputs->values[i * 4 + j] = (real)(i * 4 + j);
        for (int j = 0; j < 2; j++)
            outputs->values[i * 2 + j] = (real)(i * 2 + j);
    }

    DataSplit split;
    int result_code = hypertune_split_data(inputs, outputs, 0.8f, 0, 0, &split);
    TESTEX("split_data returns ERR_OK", (result_code == ERR_OK));
    TESTEX("train_rows is 80", (split.train_rows == 80));
    TESTEX("val_rows is 20", (split.val_rows == 20));
    TESTEX("train_inputs allocated", (split.train_inputs != NULL));
    TESTEX("train_outputs allocated", (split.train_outputs != NULL));
    TESTEX("val_inputs allocated", (split.val_inputs != NULL));
    TESTEX("val_outputs allocated", (split.val_outputs != NULL));
    TESTEX("train_inputs has correct rows", (split.train_inputs->rows == 80));
    TESTEX("val_inputs has correct rows", (split.val_inputs->rows == 20));

    hypertune_free_split(&split);
    TESTEX("free_split clears train_inputs", (split.train_inputs == NULL));

    // Test with shuffle
    result_code = hypertune_split_data(inputs, outputs, 0.8f, 1, 42, &split);
    TESTEX("split_data with shuffle returns ERR_OK", (result_code == ERR_OK));
    hypertune_free_split(&split);

    // Test error cases
    result_code = hypertune_split_data(NULL, outputs, 0.8f, 0, 0, &split);
    TESTEX("split_data with NULL inputs returns ERR_NULL_PTR", (result_code == ERR_NULL_PTR));

    result_code = hypertune_split_data(inputs, outputs, 0.0f, 0, 0, &split);
    TESTEX("split_data with ratio=0 returns ERR_INVALID", (result_code == ERR_INVALID));

    result_code = hypertune_split_data(inputs, outputs, 1.0f, 0, 0, &split);
    TESTEX("split_data with ratio=1 returns ERR_INVALID", (result_code == ERR_INVALID));

    tensor_free(inputs);
    tensor_free(outputs);

    // ========================================================================
    SUITE("Network Creation from Config");
    COMMENT("Testing network creation from HypertuneResult...");

    HypertuneResult config;
    hypertune_result_init(&config);
    config.learning_rate = 0.01f;
    config.batch_size = 32;
    config.optimizer = OPT_ADAM;
    config.hidden_layer_count = 2;
    config.hidden_layer_sizes[0] = 64;
    config.hidden_layer_sizes[1] = 32;
    config.hidden_activations[0] = ACTIVATION_RELU;
    config.hidden_activations[1] = ACTIVATION_RELU;
    config.topology_pattern = TOPOLOGY_CONSTANT;

    PNetwork net = hypertune_create_network(&config, 10, 3, ACTIVATION_SOFTMAX, LOSS_CATEGORICAL_CROSS_ENTROPY);
    TESTEX("create_network returns non-NULL", (net != NULL));
    TESTEX("Network has 4 layers (input + 2 hidden + output)", (ann_get_layer_count(net) == 4));
    TESTEX("Input layer has 10 nodes", (ann_get_layer_nodes(net, 0) == 10));
    TESTEX("First hidden layer has 64 nodes", (ann_get_layer_nodes(net, 1) == 64));
    TESTEX("Second hidden layer has 32 nodes", (ann_get_layer_nodes(net, 2) == 32));
    TESTEX("Output layer has 3 nodes", (ann_get_layer_nodes(net, 3) == 3));

    ann_free_network(net);

    // ========================================================================
    SUITE("Small Grid Search");
    COMMENT("Testing grid search with minimal configuration...");

    // Create simple XOR-like data
    real xor_inputs[] = {0,0, 0,1, 1,0, 1,1, 0,0, 0,1, 1,0, 1,1};
    real xor_outputs[] = {0, 1, 1, 0, 0, 1, 1, 0};
    
    PTensor xor_in = tensor_create_from_array(8, 2, xor_inputs);
    PTensor xor_out = tensor_create_from_array(8, 1, xor_outputs);

    // Split data (use all for training in this tiny test)
    DataSplit xor_split;
    result_code = hypertune_split_data(xor_in, xor_out, 0.75f, 0, 0, &xor_split);
    TESTEX("XOR data split OK", (result_code == ERR_OK));

    // Minimal search space
    HyperparamSpace mini_space;
    hypertune_space_init(&mini_space);
    mini_space.learning_rate_min = 0.1f;
    mini_space.learning_rate_max = 0.1f;
    mini_space.learning_rate_steps = 1;
    mini_space.batch_sizes[0] = 4;
    mini_space.batch_size_count = 1;
    mini_space.optimizers[0] = OPT_SGD;
    mini_space.optimizer_count = 1;
    mini_space.hidden_layer_counts[0] = 1;
    mini_space.hidden_layer_count_options = 1;
    mini_space.hidden_layer_sizes[0] = 4;
    mini_space.hidden_layer_size_count = 1;
    mini_space.hidden_activations[0] = ACTIVATION_SIGMOID;
    mini_space.hidden_activation_count = 1;
    mini_space.epoch_limit = 100;  // Quick training

    HypertuneOptions opts;
    hypertune_options_init(&opts);
    opts.verbosity = 0;  // Quiet for tests

    HypertuneResult results[10];
    HypertuneResult best;
    
    int trials = hypertune_grid_search(
        &mini_space, 2, 1, ACTIVATION_SIGMOID, LOSS_MSE,
        &xor_split, &opts, results, 10, &best);

    TESTEX("Grid search returns 1 trial", (trials == 1));
    TESTEX("Best result has valid score", (best.score >= 0.0f));
    TESTEX("Best result has trial_id=1", (best.trial_id == 1));

    hypertune_free_split(&xor_split);
    tensor_free(xor_in);
    tensor_free(xor_out);

    // ========================================================================
    SUITE("Random Search");
    COMMENT("Testing random search...");

    // Recreate XOR data
    xor_in = tensor_create_from_array(8, 2, xor_inputs);
    xor_out = tensor_create_from_array(8, 1, xor_outputs);
    result_code = hypertune_split_data(xor_in, xor_out, 0.75f, 0, 0, &xor_split);

    // Search with 3 random trials
    trials = hypertune_random_search(
        &mini_space, 3, 2, 1, ACTIVATION_SIGMOID, LOSS_MSE,
        &xor_split, &opts, results, 10, &best);

    TESTEX("Random search returns 3 trials", (trials == 3));
    TESTEX("Random search best has valid score", (best.score >= 0.0f));

    hypertune_free_split(&xor_split);
    tensor_free(xor_in);
    tensor_free(xor_out);

    // ========================================================================
    SUITE("Topology Generation");
    COMMENT("Testing topology pattern generation...");

    int sizes[4];
    
    // Test CONSTANT topology
    hypertune_generate_topology(TOPOLOGY_CONSTANT, 64, 3, sizes);
    TESTEX("CONSTANT: layer 0 = 64", (sizes[0] == 64));
    TESTEX("CONSTANT: layer 1 = 64", (sizes[1] == 64));
    TESTEX("CONSTANT: layer 2 = 64", (sizes[2] == 64));
    
    // Test PYRAMID topology (decreasing)
    hypertune_generate_topology(TOPOLOGY_PYRAMID, 64, 3, sizes);
    TESTEX("PYRAMID: layer 0 = 64", (sizes[0] == 64));
    TESTEX("PYRAMID: layer 1 = 32", (sizes[1] == 32));
    TESTEX("PYRAMID: layer 2 = 16", (sizes[2] == 16));
    
    // Test INVERSE topology (increasing)
    hypertune_generate_topology(TOPOLOGY_INVERSE, 64, 3, sizes);
    TESTEX("INVERSE: layer 0 = 16", (sizes[0] == 16));
    TESTEX("INVERSE: layer 1 = 32", (sizes[1] == 32));
    TESTEX("INVERSE: layer 2 = 64", (sizes[2] == 64));
    
    // Test FUNNEL topology (expand then contract)
    hypertune_generate_topology(TOPOLOGY_FUNNEL, 32, 3, sizes);
    TESTEX("FUNNEL: layer 0 expands", (sizes[0] >= 32));
    TESTEX("FUNNEL: layer 1 is larger", (sizes[1] >= sizes[0]));
    TESTEX("FUNNEL: layer 2 contracts", (sizes[2] <= sizes[1]));
    
    // Test topology names
    TESTEX("CONSTANT name", (strcmp(hypertune_topology_name(TOPOLOGY_CONSTANT), "CONSTANT") == 0));
    TESTEX("PYRAMID name", (strcmp(hypertune_topology_name(TOPOLOGY_PYRAMID), "PYRAMID") == 0));
    TESTEX("FUNNEL name", (strcmp(hypertune_topology_name(TOPOLOGY_FUNNEL), "FUNNEL") == 0));
    TESTEX("INVERSE name", (strcmp(hypertune_topology_name(TOPOLOGY_INVERSE), "INVERSE") == 0));

    // ========================================================================
    SUITE("Topology Search Integration");
    COMMENT("Testing grid search with multiple topology patterns...");

    HyperparamSpace topo_space;
    hypertune_space_init(&topo_space);
    
    // Minimal configuration with multiple topologies
    topo_space.learning_rate_min = 0.01f;
    topo_space.learning_rate_max = 0.01f;
    topo_space.learning_rate_steps = 1;
    topo_space.batch_sizes[0] = 32;
    topo_space.batch_size_count = 1;
    topo_space.optimizers[0] = OPT_ADAM;
    topo_space.optimizer_count = 1;
    topo_space.hidden_layer_counts[0] = 2;
    topo_space.hidden_layer_count_options = 1;
    topo_space.hidden_layer_sizes[0] = 16;
    topo_space.hidden_layer_size_count = 1;
    topo_space.hidden_activations[0] = ACTIVATION_SIGMOID;
    topo_space.hidden_activation_count = 1;
    topo_space.epoch_limit = 50;
    
    // Test with 3 topology patterns
    topo_space.topology_patterns[0] = TOPOLOGY_CONSTANT;
    topo_space.topology_patterns[1] = TOPOLOGY_PYRAMID;
    topo_space.topology_patterns[2] = TOPOLOGY_INVERSE;
    topo_space.topology_pattern_count = 3;
    
    // Count should be 3 trials (one per topology)
    int topo_trials = hypertune_count_grid_trials(&topo_space);
    TESTEX("Topology search: 3 trials expected", (topo_trials == 3));

    // Run grid search
    xor_in = tensor_create_from_array(8, 2, xor_inputs);
    xor_out = tensor_create_from_array(8, 1, xor_outputs);
    hypertune_split_data(xor_in, xor_out, 0.75f, 0, 0, &xor_split);
    
    opts.verbosity = 0;  // quiet
    trials = hypertune_grid_search(
        &topo_space, 2, 1, ACTIVATION_SIGMOID, LOSS_MSE,
        &xor_split, &opts, results, 10, &best);
    
    TESTEX("Topology grid search returns 3 trials", (trials == 3));
    TESTEX("Result 0 is CONSTANT", (results[0].topology_pattern == TOPOLOGY_CONSTANT));
    TESTEX("Result 1 is PYRAMID", (results[1].topology_pattern == TOPOLOGY_PYRAMID));
    TESTEX("Result 2 is INVERSE", (results[2].topology_pattern == TOPOLOGY_INVERSE));
    
    // Verify layer sizes differ by topology
    TESTEX("CONSTANT: both layers same size", (results[0].hidden_layer_sizes[0] == results[0].hidden_layer_sizes[1]));
    TESTEX("PYRAMID: layer 0 > layer 1", (results[1].hidden_layer_sizes[0] > results[1].hidden_layer_sizes[1]));
    TESTEX("INVERSE: layer 0 < layer 1", (results[2].hidden_layer_sizes[0] < results[2].hidden_layer_sizes[1]));
    
    hypertune_free_split(&xor_split);
    tensor_free(xor_in);
    tensor_free(xor_out);

    // ========================================================================
    SUITE("Gaussian Process");
    COMMENT("Testing GP prediction and expected improvement...");

    GaussianProcess gp;
    gp_init(&gp, 1);  // 1D for simple testing
    
    TESTEX("GP init: n_observations = 0", (gp.n_observations == 0));
    TESTEX("GP init: n_dims = 1", (gp.n_dims == 1));
    
    // Add some observations
    real x1[2] = {0.0f, 0.0f};
    real x2[2] = {0.5f, 0.0f};
    real x3[2] = {1.0f, 0.0f};
    
    gp_add_observation(&gp, x1, 0.5f);
    gp_add_observation(&gp, x2, 0.9f);  // best point
    gp_add_observation(&gp, x3, 0.6f);
    
    TESTEX("GP has 3 observations", (gp.n_observations == 3));
    
    // Predict at observed point - should be close to observed value
    real mean, variance;
    gp_predict(&gp, x2, &mean, &variance);
    TESTEX("GP predict at x2: mean close to 0.9", (fabs(mean - 0.9f) < 0.1f));
    TESTEX("GP predict at x2: low variance", (variance < 0.1f));
    
    // Predict at unobserved point - should have higher variance
    real x_new[2] = {0.25f, 0.0f};
    gp_predict(&gp, x_new, &mean, &variance);
    TESTEX("GP predict at new point: has variance", (variance > 0.0f));
    
    // Test Expected Improvement
    real ei = gp_expected_improvement(0.8f, 0.1f, 0.9f, 0.01f);
    TESTEX("EI for point below best: small", (ei >= 0.0f));
    
    real ei2 = gp_expected_improvement(1.0f, 0.1f, 0.9f, 0.01f);
    TESTEX("EI for point above best: larger", (ei2 > ei));

    // ========================================================================
    SUITE("Bayesian Optimization");
    COMMENT("Testing Bayesian search with minimal configuration...");

    // Create XOR data for Bayesian optimization test
    xor_in = tensor_create_from_array(8, 2, xor_inputs);
    xor_out = tensor_create_from_array(8, 1, xor_outputs);
    hypertune_split_data(xor_in, xor_out, 0.75f, 0, 0, &xor_split);
    
    // Minimal search space
    HyperparamSpace bo_space;
    hypertune_space_init(&bo_space);
    bo_space.learning_rate_min = 0.01f;
    bo_space.learning_rate_max = 0.5f;
    bo_space.batch_sizes[0] = 4;
    bo_space.batch_sizes[1] = 8;
    bo_space.batch_size_count = 2;
    bo_space.optimizers[0] = OPT_SGD;
    bo_space.optimizer_count = 1;
    bo_space.hidden_layer_counts[0] = 1;
    bo_space.hidden_layer_count_options = 1;
    bo_space.hidden_layer_sizes[0] = 4;
    bo_space.hidden_layer_size_count = 1;
    bo_space.hidden_activations[0] = ACTIVATION_SIGMOID;
    bo_space.hidden_activation_count = 1;
    bo_space.epoch_limit = 50;
    
    // Bayesian options - small for testing
    BayesianOptions bo_opts;
    bayesian_options_init(&bo_opts);
    bo_opts.n_initial = 3;
    bo_opts.n_iterations = 2;
    bo_opts.n_candidates = 20;
    
    opts.verbosity = 0;
    
    trials = hypertune_bayesian_search(
        &bo_space, 2, 1, ACTIVATION_SIGMOID, LOSS_MSE,
        &xor_split, &opts, &bo_opts, results, 10, &best);
    
    TESTEX("Bayesian search returns 5 trials (3 init + 2 BO)", (trials == 5));
    TESTEX("Bayesian search best has valid score", (best.score >= 0.0f));
    TESTEX("All results have learning rates in range", 
           (results[0].learning_rate >= 0.01f && results[0].learning_rate <= 0.5f));
    
    hypertune_free_split(&xor_split);
    tensor_free(xor_in);
    tensor_free(xor_out);

    // ========================================================================
    SUITE("TPE Initialization");
    COMMENT("Testing TPE options initialization...");

    TPEOptions tpe_opts;
    tpe_options_init(&tpe_opts);
    TESTEX("TPE init: n_startup = 10", (tpe_opts.n_startup == 10));
    TESTEX("TPE init: gamma = 0.25", (fabs(tpe_opts.gamma - 0.25f) < 0.01f));
    TESTEX("TPE init: n_candidates = 24", (tpe_opts.n_candidates == 24));
    TESTEX("TPE init: n_iterations = 40", (tpe_opts.n_iterations == 40));
    TESTEX("TPE init: bandwidth_factor = 1.0", (fabs(tpe_opts.bandwidth_factor - 1.0f) < 0.01f));

    // ========================================================================
    SUITE("TPE Search");
    COMMENT("Testing TPE search with minimal configuration...");

    // Create XOR data for TPE test
    xor_in = tensor_create_from_array(8, 2, xor_inputs);
    xor_out = tensor_create_from_array(8, 1, xor_outputs);
    hypertune_split_data(xor_in, xor_out, 0.75f, 0, 0, &xor_split);
    
    // Search space with multiple categorical options
    HyperparamSpace tpe_space;
    hypertune_space_init(&tpe_space);
    tpe_space.learning_rate_min = 0.01f;
    tpe_space.learning_rate_max = 0.5f;
    tpe_space.learning_rate_log_scale = 1;
    tpe_space.batch_sizes[0] = 4;
    tpe_space.batch_sizes[1] = 8;
    tpe_space.batch_size_count = 2;
    tpe_space.optimizers[0] = OPT_SGD;
    tpe_space.optimizers[1] = OPT_ADAM;
    tpe_space.optimizer_count = 2;
    tpe_space.hidden_layer_counts[0] = 1;
    tpe_space.hidden_layer_counts[1] = 2;
    tpe_space.hidden_layer_count_options = 2;
    tpe_space.hidden_layer_sizes[0] = 4;
    tpe_space.hidden_layer_size_count = 1;
    tpe_space.hidden_activations[0] = ACTIVATION_SIGMOID;
    tpe_space.hidden_activations[1] = ACTIVATION_RELU;
    tpe_space.hidden_activation_count = 2;
    tpe_space.epoch_limit = 50;
    
    // TPE options - small for testing
    tpe_options_init(&tpe_opts);
    tpe_opts.n_startup = 3;
    tpe_opts.n_iterations = 2;
    tpe_opts.n_candidates = 10;
    
    opts.verbosity = 0;
    
    trials = hypertune_tpe_search(
        &tpe_space, 2, 1, ACTIVATION_SIGMOID, LOSS_MSE,
        &xor_split, &opts, &tpe_opts, results, 10, &best);
    
    TESTEX("TPE search returns 5 trials (3 startup + 2 TPE)", (trials == 5));
    TESTEX("TPE search best has valid score", (best.score >= 0.0f));
    TESTEX("TPE results have learning rates in range", 
           (results[0].learning_rate >= 0.01f && results[0].learning_rate <= 0.5f));
    TESTEX("TPE results have valid batch sizes", 
           (results[0].batch_size == 4 || results[0].batch_size == 8));
    TESTEX("TPE results have valid optimizers", 
           (results[0].optimizer == OPT_SGD || results[0].optimizer == OPT_ADAM));
    
    hypertune_free_split(&xor_split);
    tensor_free(xor_in);
    tensor_free(xor_out);
}
