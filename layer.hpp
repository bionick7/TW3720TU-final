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
