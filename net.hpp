#include "layer.hpp"

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
