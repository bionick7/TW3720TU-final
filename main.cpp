#include "layer.hpp"

// Function to calculate the loss
template <typename T>
T MSEloss(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
    // Your implementation of the MSEloss function starts here
};

// Function to calculate the gradients of the loss
template <typename T>
Matrix<T> MSEgrad(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
    // Your implementation of the MSEgrad function starts here
}

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

int main(int argc, char* argv[])
{
    // Your training and testing of the Net class starts here

    printf("Hello world!\n");

    return 0;
}