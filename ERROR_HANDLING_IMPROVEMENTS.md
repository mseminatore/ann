# Error Handling Improvements

## Overview
This document summarizes the error handling improvements made to the libann codebase to improve robustness, debugging, and reliability.

## New Error Codes (ann.h)
Added granular error codes to replace the generic `ERR_FAIL`:

```c
#define ERR_FAIL        -1      // Generic failure (deprecated)
#define ERR_OK           0      // Success
#define ERR_NULL_PTR    -2      // Null pointer provided
#define ERR_ALLOC       -3      // Memory allocation failed
#define ERR_INVALID     -4      // Invalid parameter or state
#define ERR_IO          -5      // File I/O error
```

### New Macro for Null Checks
```c
#define CHECK_NULL(ptr) if ((ptr) == NULL) return ERR_NULL_PTR
```

## Improvements by Module

### tensor.c

#### 1. Input Validation
- **`tensor_create()`**: Added validation for rows/cols > 0
- **Binary operations**: Enhanced null checks with assertion messages:
  - `tensor_add()`: "tensor_add: null tensor"
  - `tensor_sub()`: "tensor_sub: null tensor"
  - `tensor_mul()`: "tensor_mul: null tensor"
  - `tensor_div()`: "tensor_div: null tensor"

### ann.c

#### 1. Layer Creation (`ann_add_layer()`)
- **Null pointer validation**: Checks for null network, invalid node counts
- **Memory leak prevention**: Comprehensive rollback on tensor allocation failures
  - Decrements layer count on failure
  - Frees all successfully allocated tensors on partial failure
  - Returns `ERR_ALLOC` on allocation failure
- **Better error codes**: Returns `ERR_NULL_PTR`, `ERR_INVALID`, `ERR_ALLOC`

**Before:**
```c
if (!pnet) return ERR_FAIL;
pnet->layers[cur_layer].t_values = tensor_zeros(1, node_count);
pnet->layers[cur_layer].node_count = node_count;
// No checks if allocation failed!
```

**After:**
```c
if (!pnet) return ERR_NULL_PTR;
if (node_count <= 0) return ERR_INVALID;
// ... allocation with comprehensive error handling
pnet->layers[cur_layer].t_values = tensor_zeros(1, node_count);
if (pnet->layers[cur_layer].t_values == NULL) {
    pnet->layer_count--;
    return ERR_ALLOC;
}
// ... handles each subsequent allocation with rollback
```

#### 2. Training (`ann_train_network()`)
- **Input validation**: Checks for null tensors, invalid dimensions
- **Data consistency checks**: Verifies input/output row counts match
- **Prevents division by zero**: Checks rows > 0

#### 3. Prediction (`ann_predict()`)
- **Null pointer validation**: Checks network, input, and output
- **Network state validation**: Verifies layer_count and layers array
- **Better error codes**: Returns `ERR_NULL_PTR`, `ERR_INVALID` instead of `ERR_FAIL`

#### 4. CSV Loading (`ann_load_csv()`)
- **Parameter validation**: Checks all pointer parameters
- **File I/O errors**: Returns `ERR_IO` instead of `ERR_FAIL` for fopen() failures
- **Better error context**: Distinguishes between null pointers and file errors

#### 5. Network Serialization
- **`ann_save_network_binary()`**: Validates network and filename, uses `ERR_IO` for file errors
- **`ann_save_network()`**: Same improvements as binary version

#### 6. Network Properties
- **`ann_evaluate_accuracy()`**: Added dimension validation
  - Checks tensor dimensions are valid
  - Verifies row counts match between inputs and outputs
  - Returns -1.0 on validation failure

## Benefits

### Immediate Benefits
✅ **Better debugging**: Specific error codes indicate root cause
✅ **Memory safety**: Prevents resource leaks on allocation failures
✅ **Input validation**: Catches invalid parameters early
✅ **Data integrity**: Validates dimension consistency

### Long-term Benefits
✅ **Easier maintenance**: Clearer error conditions for future developers
✅ **More robust error handling**: Graceful degradation instead of silent failures
✅ **Better separation of concerns**: Different error codes for different failure modes
✅ **Easier logging/monitoring**: Distinct error types for filtering/alerting

## Compatibility

**Backward Compatible**: Legacy code using `ERR_OK` and `ERR_FAIL` continues to work:
- `ERR_OK` (0) remains success indicator
- `ERR_FAIL` (-1) is still returned for general failures
- New code should use specific error codes

## Testing

✅ All existing tests pass
✅ No performance impact
✅ No API changes (return types unchanged)
✅ All improvements are internal

## Future Recommendations

1. **Add error message helper**: Create `const char* ann_strerror(int error_code)`
2. **Add logging**: Optional error logging callback during initialization
3. **Expand validation**: Consider adding optional parameter bounds checking
4. **Error recovery**: Implement cleanup handlers for automatic resource management
5. **Unit tests**: Create dedicated error handling test suite
