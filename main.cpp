#include "net.hpp"

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
Matrix<T> argmax(const Matrix<T>& y) 
{
    // Your implementation of the argmax function starts here
}

// Calculate the accuracy of the prediction, using the argmax
template <typename T>

T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{
    // Your implementation of the get_accuracy starts here
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

        std::cout << "Step " << step << ", Loss: " << loss << std::endl;

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