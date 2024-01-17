#include <cmath>
#include <initializer_list>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>

#include <sstream>
template <typename T>
class Matrix {

    // To access elemets of Matrices with different template parameters
    template<class U>
    friend class Matrix;

    int rows, columns;
    T* elements;

public:

    Matrix() {
        _allocate(0, 0);
    }

    Matrix(int p_rows, int p_columns) {
        _allocate(p_rows, p_columns);
        for(int i=0; i < rows * columns; i++) {
            elements[i] = 0;
        }
    }

    Matrix(int p_rows, int p_columns, const std::initializer_list<T>& list) {
        _allocate(p_rows, p_columns);
        int i=0;
        if ((int) list.size() != p_rows * p_columns) {
            throw std::range_error("Matrix initializer list must have the same length as matrix size (rows x columns)");
        }
        for (auto iter = list.begin(); iter != list.end();  iter++){
            elements[i++] = *iter;
        }
    }

    Matrix(const Matrix<T>& other) {
        _allocate(other.rows, other.columns);
        for(int i=0; i < rows * columns; i++) {
            elements[i] = other.elements[i];
        }
    }

    Matrix(Matrix<T>&& other) {
        rows = other.rows;
        columns = other.columns;
        elements = other.elements;
        other.rows = 0;
        other.columns = 0;
        other.elements = nullptr;
    }

    ~Matrix() {
        _deallocate();
    }

    Matrix<T>& operator=(const Matrix<T>& other) {
        if (other.rows*other.columns == rows*columns) {
            // Can skip reallocation
            rows = other.rows;
            columns = other.columns;
        } else {
            _deallocate();
            _allocate(other.rows, other.columns);
        }
        for(int i=0; i < rows * columns; i++) {
            elements[i] = other.elements[i];
        }
        return *this;
    }

    Matrix<T>& operator=(Matrix<T>&& other) {
        rows = other.rows;
        columns = other.columns;
        elements = other.elements;
        other.rows = 0;
        other.columns = 0;
        other.elements = nullptr;
        return *this;
    }

    // /= operator
    Matrix<T>& operator/=(const T& scalar) {
        if (scalar == 0) {
            throw std::domain_error("Division by zero is not allowed.");
        }

        for (int i = 0; i < rows * columns; i++) {
            elements[i] /= scalar;
        }

        return *this;
    }

    T& operator[](const std::pair<int, int>& ij) {
        _assertInside(ij.first, ij.second);
        return elements[ij.first * columns + ij.second];
    }

    const T& operator[](const std::pair<int, int>& ij) const {
        _assertInside(ij.first, ij.second);
        return elements[ij.first * columns + ij.second];
    }

    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(U x) const {
        auto res = Matrix<typename std::common_type<T,U>::type>(rows, columns);
        // Slightly inefficient since we are forced to initialize with 0;
        for(int i=0; i < rows*columns; i++) {
            res.elements[i] = elements[i] * x;
        }
        return res;
    }

    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(const Matrix<U>& B) const {
        if (columns != B.rows) {
            std::stringstream error_message;
            error_message << "Cannot multiply matrix of dimension (" << rows << " x " << columns
                          << "), with one of dimension" << "(" << B.rows << " x " << B.columns << ")";
            throw std::domain_error(error_message.str());
        }
        // 0 initialization is necaissary here;
        auto res = Matrix<typename std::common_type<T,U>::type>(rows, B.columns);
        for(int i=0; i < res.rows*res.columns; i++) {
            int row = i / res.columns;
            int col = i % res.columns;
            for(int j = 0; j < columns; j++) {  // Dot product
                //printf("%f * %f + ", elements[row * columns + j], B.elements[j * B.columns + col]);
                res.elements[i] += elements[row * columns + j] * B.elements[j * B.columns + col];
            }
            //printf("\n");
        }
        return res;
    }

    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator+(const Matrix<U>& B) const {
        if (B.columns != columns) {
            throw std::domain_error("Cannot add matrices with different columns");
        }
        if (B.rows != rows && B.rows != 1 && rows != 1) {
            throw std::domain_error("Cannot add matrices with different rows other than rows = 1");
        }
        int res_rows = rows > B.rows ? rows : B.rows;
        auto res = Matrix<typename std::common_type<T,U>::type>(res_rows, columns);

        // Special case: we have only 1 row
        if (rows == 1) {
            for(int i=0; i < res_rows*columns; i++) {
                res.elements[i] = elements[i % res.columns] + B.elements[i];
            }
            return res;
        }

        // Special case: B has only 1 row
        if (B.rows == 1) {
            for(int i=0; i < res_rows*columns; i++) {
                res.elements[i] = elements[i] + B.elements[i % res.columns];
            }
            return res;
        }

        // General case
        for(int i=0; i < res_rows*columns; i++) {
            res.elements[i] = elements[i] + B.elements[i];
        }

        return res;
    }

    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator-(const Matrix<U>& B) const {
        return *this + (B * -1);
    }

    Matrix transpose() const {
        Matrix res = Matrix(columns, rows);
        for(int i=0; i < rows*columns; i++) {
            int row = i / res.columns;
            int col = i % res.columns;
            res.elements[i] = elements[col * columns + row];
        }
        return res;
    }

    int getRows() const {
        return rows;
    }

    int getCols() const {
        return columns;
    }

    T sum() const {
        T result = 0;
        for (int i = 0; i < rows * columns; ++i) {
            result += elements[i];
        }
        return result;
    }

    // Modify the sumRows method
    template<typename U = T>
    Matrix<typename std::common_type<T,U>::type> sumRows() const {
        auto result = Matrix<typename std::common_type<T,U>::type>(1, columns);

        for (int j = 0; j < columns; ++j) {
            typename std::common_type<T,U>::type sum = 0;
            for (int i = 0; i < rows; ++i) {
                sum += elements[i * columns + j];
            }
            result[{0, j}] = sum;
        }
        
        return result;
    }


    // Debugging methods -- not asked for, but usefull
    void inspect(bool printData = false) const {
        if (rows * columns == 0) {
            std::cout << " [ ]";
            return;
        }

        std::cout << "Matrix Size: " << rows << "x" << columns << std::endl;

        if (printData) {
            std::cout << " [";
            std::cout << elements[0];
            for (int i = 1; i < rows * columns; i++) {
                if (i % columns == 0) {
                    std::cout << "\n  " << elements[i];
                } else {
                    std::cout << ",  " << elements[i];
                }
            }
            std::cout << "]" << std::endl;
        };
    }


private:

    // Initializes the memory etc. without filling it. Used in multiple constructors
    void _allocate(int p_rows, int p_columns) {
        rows = p_rows;
        columns = p_columns;
        if (rows * columns == 0) {
            // Not sure how T[0] behaves, so let's be explicit
            elements = nullptr;
        } else {
            elements = new T[rows * columns];
        }
    }

    // Destructor. Also needed in e.g. copy operator
    void _deallocate() {
        // Will work finn for nullptr
        delete[] elements;
    }

    // asserts if index is inside of bounds
    void _assertInside(int row_index, int column_index) const {
        if (row_index >= rows || column_index >= columns) {
            std::stringstream error_message;
            error_message << "Tried to access (" << row_index << " x " << column_index 
                          << "), which is outside the bound of the matrix with size"
                          << "(" << rows << " x " << columns << ")";
            throw std::range_error(error_message.str());
        }
    }
};

// can't use assert, so I use this. 
// Benefit of macros is putting the condition into the error message as a string
#define CUSTOM_ASSERT(condition) \
if (!(condition)) { \
    std::stringstream error_message;\
    error_message << __FILE__ << ":" << __LINE__ << " Assert fail: '" #condition "' not true"; \
    throw std::runtime_error(error_message.str());\
}
#define ASSERT_ERROR(operation) \
    { \
        bool failed = false; \
        try {  \
            operation; \
        } catch (const std::exception& e) { \
            failed = true; \
        } \
        if(!failed) { \
            throw std::runtime_error("Expected: '" #operation "' to fail"); \
        } \
    }

void matrixTests() {
    // Test constructors and assignment
    Matrix<int> m_empty = Matrix<int>();
    Matrix<int> m1 = Matrix<int>(2, 2, {1, 2, 3, 4});
    printf("Constructors:\n");
    for(int i=0; i < 4; i++) {
        //printf("%d => %d\n", i, m1[{i/2, i%2}]);
        CUSTOM_ASSERT((m1[{i/2, i%2}] == i+1))
    }
    ASSERT_ERROR(Matrix<int> m2 = Matrix<int>(2, 2, {1, 2, 3, 4, 5}))
    ASSERT_ERROR(Matrix<int> m2 = Matrix<int>(2, 2, {1}))
    Matrix<int> m4 = Matrix<int>(2, 2);
    for(int i=0; i < 4; i++) {
        CUSTOM_ASSERT((m4[{i%2, i/2}] == 0))
    }
    Matrix<int> m1_copy = m1;
    Matrix<int> m1_copy2 = Matrix<int>(3, 3);
    m1_copy2 = m1;
    printf("Copy:\n");
    for(int i=0; i < 4; i++) {
        CUSTOM_ASSERT((m1[{i/2, i%2}] == m1_copy[{i/2, i%2}]))
        CUSTOM_ASSERT((m1[{i/2, i%2}] == m1_copy2[{i/2, i%2}]))
    }
    Matrix<int> m1_move = std::move(Matrix<int>(m1));
    Matrix<int> m1_move2 = Matrix<int>(3, 3);
    m1_move2 = std::move(m1);

    printf("Move:\n");
    for(int i=0; i < 4; i++) {
        CUSTOM_ASSERT((m1_move[{i/2, i%2}] == m1_copy[{i/2, i%2}]))
    }

    // Test operators

    ASSERT_ERROR((m1[{2, 1}]))

    // 0.0 1.1
    // 1.1 0.0
    auto a = Matrix<double>(2, 2, {0.0, 1.1, 1.1, 0.0});

    auto bias_3 = Matrix<double>(1, 3, {1, 2, 3});

    // 0.0 1.1
    //-2.1 1.0
    auto b = Matrix<double>(2, 2, {0.0, 1.1, -2.1, 1.0});
    auto c = Matrix<double>(3, 3);
    // 0.0 1.1 5.0
    //-2.1 1.0 0.0
    auto d = Matrix<double>(2, 3, {0.0, 1.1, 5.0, -2.1, 1.0, 0.0});
    //-2.31  1.1   0.0
    //-2.1 -1.31 -10.5
    auto bxd_expected = Matrix<double>(2, 3, {-2.31, 1.1, 0.0, -2.1, -1.31, -10.5});
    auto a_plus_b_expected = Matrix<double>(2, 2, {0.0, 2.2, -1.0, 0.0});

    printf("Addition:\n");
    ASSERT_ERROR(a + c)
    ASSERT_ERROR(a + d)
    ASSERT_ERROR(d + a)
    auto a_plus_b = a + b;
    CUSTOM_ASSERT(a_plus_b.getRows() == 2);
    CUSTOM_ASSERT(a_plus_b.getCols() == 2);
    for(int i=0; i < 2; i++) {
        CUSTOM_ASSERT((a_plus_b[{i/2, i%2}] == a_plus_b_expected[{i/2, i%2}]))
    }

    auto d_plus_bias = d + bias_3;
    auto bias_plus_d = bias_3 + d;
    CUSTOM_ASSERT(d_plus_bias.getRows() == d.getRows());
    CUSTOM_ASSERT(d_plus_bias.getCols() == d.getCols());
    CUSTOM_ASSERT(bias_plus_d.getRows() == d.getRows());
    CUSTOM_ASSERT(bias_plus_d.getCols() == d.getCols());

    for(int i=0; i < 6; i++) {
        CUSTOM_ASSERT(fabs(d_plus_bias[{i/3, i%3}] - bias_plus_d[{i/3, i%3}]) < 1e-5)
        CUSTOM_ASSERT(fabs(d_plus_bias[{i/3, i%3}] - d[{i/3, i%3}] - bias_3[{0, i%3}]) < 1e-5)
    }

    printf("Multiplication:\n");
    auto d_scale_5 = d * 5ul;
    for(int i=0; i < 6; i++) {
        CUSTOM_ASSERT(fabs(d_scale_5[{i/3, i%3}] - d[{i/3, i%3}] * 5) < 1e-5)
    }

    printf("Substration:\n");
    ASSERT_ERROR(a - c)
    ASSERT_ERROR(a - d)
    ASSERT_ERROR(d - a)
    auto a_minus_b = a - b;
    auto b_minus_a = b - a;
    for(int i=0; i < 2; i++) {
        CUSTOM_ASSERT(fabs(a_minus_b[{i/2, i%2}] + b_minus_a[{i/2, i%2}]) < 1e-5)
    }

    printf("Transpose:\n");
    auto d_tt = d.transpose().transpose();
    CUSTOM_ASSERT(((a + b)[{1, 0}] == -1.0))

    for(int i=0; i < 6; i++) {
        CUSTOM_ASSERT((d_tt[{i/3, i%3}] == d[{i/3, i%3}]))
    }
}
template<typename T>
class Layer
{
public:
    // shared attributes
    int in_features;
    int out_features;
    int n_samples;

    // Virtual forward
    virtual Matrix<T> forward(const Matrix<T>& x) = 0;

    // Virtual backward
    virtual Matrix<T> backward(const Matrix<T>& dy) = 0;
};


template<typename T>
class Linear: public Layer<T> 
{
public:
    // Attributes
    Matrix<T> cache;
    Matrix<T> bias;
    Matrix<T> weights;
    Matrix<T> bias_gradients;
    Matrix<T> weights_gradients;

    // Constructor
    Linear(int in_features, int out_features, int n_samples, int seed) 
    {
        // initialize attributes
        this->in_features = in_features;
        this->out_features = out_features;
        this->n_samples = n_samples;
        
        //initialize rng
        std::default_random_engine generator(seed);
        std::normal_distribution<T> distribution_normal(0.0, 1.0);
        std::uniform_real_distribution<T> distribution_uniform(0.0, 1.0);
        
        // initialize matrices
        cache = Matrix<T>(n_samples, in_features);
        
        this->bias = Matrix<T>(1, out_features);
        
        for (int j=0; j<out_features; ++j) {
                bias[{0,j}] = distribution_uniform(generator);
        }
                
                this->weights = Matrix<T>(in_features, out_features);
                
        for (int i=0; i<in_features; ++i) {
            for (int j=0; j<out_features; ++j) {
                this->weights[{i,j}] = distribution_normal(generator);
            }
        }    
        
        this->bias_gradients = Matrix<T>(1, out_features);
        this->weights_gradients = Matrix<T>(in_features, out_features);
        
    };
    
    // Destructor
    ~Linear() {};
    
    // Member functions
    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        // Store input in cache
        cache = x;
        
        // Return result
        return x*this->weights+this->bias;
    };
    
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        // Compute gradients during backward pass
        this->weights_gradients = this->cache.transpose() * dy;
        this->bias_gradients = dy.sumRows();

        // Return result
        return dy*this->weights.transpose();
    };
    
    void optimize(T learning_rate) {
        // Update weights and bias
        this->weights = this->weights - this->weights_gradients*learning_rate;
        this->bias = this->bias - this->bias_gradients*learning_rate;
    };

    void printWeights(bool printData = false) const {
        std::cout << "Weights:\n";
        weights.inspect(printData);
        std::cout << "\nBias:\n";
        bias.inspect(printData);
        std::cout << std::endl;
    }
    
};


template<typename T>
class ReLU: public Layer<T>
{
public:
    // Member attributes
    Matrix<T> cache;

    // Constructor
    ReLU(int in_features, int out_features, int n_samples) {
        // initialize attributes
        this->in_features = in_features;
        this->out_features = out_features;
        this->n_samples = n_samples;
        
        // initialize matrices
        cache = Matrix<T>(n_samples, in_features);
    };
    
    // Destructor
    ~ReLU() {};
    
    // Member Functions
    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        // initialize matrix
        cache = x;
        
        Matrix<T> result = Matrix<T>(x.getRows(), x.getCols());
        
        // For loop to fill in matrix
        for (int i=0; i<x.getRows(); i++) {
            for (int j=0; j<x.getCols(); j++) {
                result[{i,j}] = std::max<T>(0,x[{i,j}]);
            }
        }
        
        return result;
    };
    
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        // Initialize result
        Matrix<T> result = Matrix<T>(cache.getRows(), cache.getCols());
        
        // For loop with derivative of ReLU
        for (int i=0; i<result.getRows(); i++) {
            for (int j=0; j<result.getCols(); j++) {
                if (cache[{i,j}]>0) {
                    // cache and dy should be same size
                    result[{i,j}] = dy[{i,j}];
                }
            }
        }
        
        return result;
    };
};

void layerTests()
{
    // Initialize Linear layer for testing
    Linear<double> linearTest(2, 2, 4, 1);

    // Print weights and biases before optimize
    std::cout << "Weights before optimize:\n";
    linearTest.printWeights(true);

    // Test the optimize function
    double learning_rate = 0.005;
    linearTest.optimize(learning_rate);

    std::cout << "\nWeights after optimize:\n";
    linearTest.printWeights(true);

}

template <typename T>
class Net 
{
public:
    Linear<T> linear1_;
    ReLU<T> relu_;
    Linear<T> linear2_;

    // Constructor
    Net(int in_features, int hidden_dim, int out_features, int n_samples, int seed)
        : linear1_(in_features, hidden_dim, n_samples, seed),
          relu_(hidden_dim, hidden_dim, n_samples),
          linear2_(hidden_dim, out_features, n_samples, seed)
    {}

    // Destructor
    ~Net() {}

    // Forward function
    Matrix<T> forward(const Matrix<T>& x) {
        Matrix<T> result = linear1_.forward(x);
        result = relu_.forward(result);
        result = linear2_.forward(result);
        return result;
    }

    // Backward function
    Matrix<T> backward(const Matrix<T>& dy) {
        Matrix<T> result = linear2_.backward(dy);
        result = relu_.backward(result);
        result = linear1_.backward(result);
        return result;
    }

    // Optimize function
    void optimize(T learning_rate) {
        linear1_.optimize(learning_rate);
        linear2_.optimize(learning_rate);
    }

};

template <typename T>
T MSEloss(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    int n_samples = y_true.getRows();
    Matrix<T> diff = y_pred - y_true;
    Matrix<T> squared_diff = diff * diff.transpose();
    T loss = squared_diff.sum() / (2 * n_samples);
    return loss;
};


template <typename T>
Matrix<T> MSEgrad(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    int n_samples = y_true.getRows();
    Matrix<T> gradient = y_pred - y_true;
    gradient /= n_samples;
    return gradient;
};

// Calculate the argmax 
template <typename T>
Matrix<T> argmax(const Matrix<T>& y) {
    int rows = y.getRows();
    int cols = y.getCols();

    Matrix<T> result(1, cols);

    for (int j = 0; j < cols; ++j) {
        T max_val = y[{0, j}];
        int max_index = 0;

        for (int i = 1; i < rows; ++i) {
            if (y[{i, j}] > max_val) {
                max_val = y[{i, j}];
                max_index = i;
            }
        }

        result[{0, j}] = max_index;
    }

    return result;
}

// Calculate the accuracy of the prediction, using the argmax
template <typename T>
T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred) {
    int cols = y_true.getCols();
    int correct_count = 0;

    Matrix<T> y_true_argmax = argmax(y_true);
    Matrix<T> y_pred_argmax = argmax(y_pred);

    // Compare the indices to calculate accuracy
    for (int j = 0; j < cols; ++j) {
        if (y_true_argmax[{0, j}] == y_pred_argmax[{0, j}]) {
            correct_count++;
        }
    }

    T accuracy = static_cast<T>(correct_count) / cols;
    return accuracy;
}


// Training loop
void training(bool use_test_and_debug) {
    std::initializer_list<double> xor_data = {0, 0, 1, 1, 0, 1, 0, 1};
    Matrix<double> x_xor(4, 2, xor_data);

    std::initializer_list<double> xor_labels = {1, 0, 0, 1, 0, 1, 1, 0};
    Matrix<double> y_xor(4, 2, xor_labels);

    double learning_rate = 0.005;
    int optimizer_steps = 100;

    int in_features = 2;
    int hidden_dim = 100;
    int out_features = 2;
    int seed = 1;
    int n_samples = 4;

    Net<double> net(in_features, hidden_dim, out_features, n_samples, seed);

    if (use_test_and_debug){
        std::cout << "Linear layer 1: " << std::endl;
        net.linear1_.printWeights();

        std::cout << "Linear layer 2: " << std::endl;
        net.linear2_.printWeights();
    }
    
    for (int step = 1; step <= optimizer_steps; ++step) {
        Matrix<double> predictions = net.forward(x_xor);
        double loss = MSEloss(y_xor, predictions);
        
        Matrix<double> gradient = MSEgrad(y_xor, predictions);

        net.backward(gradient);
        net.optimize(learning_rate);

        double accuracy = get_accuracy(y_xor, predictions);

        std::cout << "Step " << step << ", Loss: " << loss << ", Accuracy: " << accuracy << std::endl;

        if (use_test_and_debug){
            std::cout << "Linear layer 1: " << std::endl;
            net.linear1_.printWeights(true);

            std::cout << "Linear layer 2: " << std::endl;
            net.linear2_.printWeights(true);
        }
    }
};


int main(int argc, char* argv[])
{
    // debug variables
    bool use_test_and_debug = false;

    try {
        // individual unit tests
        if (use_test_and_debug){
            std::cout << "Tests:" << std::endl;
            matrixTests();
            layerTests();
        };

        // Final training loop
        std::cout << "Training:" << std::endl;
        training(use_test_and_debug);

    } catch (const std::exception& e) {
        
        printf("%s\n", e.what());
        return 1;
    }

    return 0;
}