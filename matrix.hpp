#include "general_includes.hpp"

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
        if (list.size() != p_rows * p_columns) {
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

    // Debugging methods -- not asked for, but usefull

    void inspect() const {
        if (rows*columns == 0) {
            std::cout << " [ ]";
            return;
        }
        std::cout << " [";
        std::cout << elements[0];
        for(int i=1; i < rows*columns; i++) {
            if (i % columns == 0) {
                std::cout << "\n  " << elements[i];
            } else {
                std::cout << ",  " << elements[i];
            }
        }
        std::cout << "]" << std::endl;
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
    auto a = Matrix(2, 2, {0.0, 1.1, 1.1, 0.0});

    auto bias_3 = Matrix(1, 3, {1, 2, 3});

    // 0.0 1.1
    //-2.1 1.0
    auto b = Matrix(2, 2, {0.0, 1.1, -2.1, 1.0});
    auto c = Matrix<double>(3, 3);
    // 0.0 1.1 5.0
    //-2.1 1.0 0.0
    auto d = Matrix(2, 3, {0.0, 1.1, 5.0, -2.1, 1.0, 0.0});
    //-2.31  1.1   0.0
    //-2.1 -1.31 -10.5
    auto bxd_expected = Matrix(2, 3, {-2.31, 1.1, 0.0, -2.1, -1.31, -10.5});
    auto a_plus_b_expected = Matrix(2, 2, {0.0, 2.2, -1.0, 0.0});

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