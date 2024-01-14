#include "matrix.hpp"

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
        
        bias = Matrix<T>(1, out_features);
        
        for (int j=0; j<out_features; ++j) {
                bias[{0,j}] = distribution_uniform(generator);
        }
                
        weights = Matrix<T>(in_features, out_features);
                
        for (int i=0; i<in_features; ++i) {
            for (int j=0; j<out_features; ++j) {
                weights[{i,j}] = distribution_normal(generator);
            }
        }    
        
        bias_gradients = Matrix<T>(1, out_features);
        weights_gradients = Matrix<T>(in_features, out_features);
        
    };
    
    // Destructor
    ~Linear() {};
    
    // Member functions
    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        // Store input in cache
        cache = x;
        
        // Return result
        return x*weights+bias;
    };
    
    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        // Return result
        return dy*weights.transpose();
    };
    
    void optimize(T learning_rate) {
        // Update weights and bias
        weights -= weights_gradients*learning_rate;
        bias -= bias_gradients*learning_rate;
    };
    
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
    ReLU<double> test = ReLU<double>(2, 2, 4);
    
    
    std::initializer_list<double> t_list = {-1.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0};
    Matrix<double> x_test = Matrix<double>(4, 2, t_list);
    
    // std::cout << test.bias[{0,0}] << std::endl;
    
    std::cout << test.forward(x_test)[{1,0}] << std::endl;
}