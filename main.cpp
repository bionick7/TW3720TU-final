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