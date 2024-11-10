from node import *
from executor import *
import argparse


def test_forward():
    x = Variable('x')
    y = x
    x_val = np.array([1, 2, 3])
    y_val = Executor([y]).run({x: x_val})[0]
    assert isinstance(y, Node)
    assert np.array_equal(y_val, x_val)


def test_gradient():
    x = Variable('x')
    y = x
    grad_x = gradient(y, [x])[0]

    x_val = np.array([1, 2, 3])
    y_val, x_grad_val = Executor([y, grad_x]).run({x: x_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x_val)
    assert np.array_equal(x_grad_val, np.ones_like(x_val))


def test_var_add_var():
    x1 = Variable('x1')
    x2 = Variable('x2')
    y = add_op(x1, x2)

    grad_x1, grad_x2 = gradient(y, [x1, x2])

    x1_val = np.array([2, 1])
    x2_val = np.array([1, 1])
    y_val, x1_grad_val, x2_grad_val = Executor([y, grad_x1, grad_x2]).run({x1: x1_val, x2: x2_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val + x2_val)
    assert np.array_equal(x1_grad_val, np.ones_like(x1_val))
    assert np.array_equal(x2_grad_val, np.ones_like(x2_val))


def test_var_add_const():
    x1 = Variable('x1')
    y = add_const_op(x1, 3)
    grad_x1 = gradient(y, [x1])[0]

    x1_val = np.array([1, 1])
    y_val, x1_grad_val = Executor([y, grad_x1]).run({x1: x1_val})

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val + 3)
    assert np.array_equal(x1_grad_val, np.ones_like(x1_val))


def test_var_mul_var():
    x1 = Variable('x1')
    x2 = Variable('x2')
    y = mul_op(x1, x2)

    grad_x1, grad_x2 = gradient(y, [x1, x2])

    x1_val = np.array([1, 2, 3])
    x2_val = np.array([4, 5, 6])
    y_val, x1_grad_val, x2_grad_val = Executor([y, grad_x1, grad_x2]).run({x1: x1_val, x2: x2_val})
    # print(y_val, x1_grad_val, x2_grad_val)

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val * x2_val)
    assert np.array_equal(x1_grad_val, x2_val)
    assert np.array_equal(x2_grad_val, x1_val)


def test_var_mul_const():
    x1 = Variable('x1')
    y = mul_const_op(x1, 4)
    grad_x1 = gradient(y, [x1])[0]
    x1_val = np.array([1, 2, 3])
    y_val, x1_grad_val = Executor([y, grad_x1]).run({x1: x1_val})
    # print(y_val,x1_grad_val)

    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val * 4)
    assert np.array_equal(x1_grad_val, np.full_like(x1_val, 4))


def test_add_mul_mix_1():
    x1 = Variable('x1')
    x2 = Variable('x2')
    # y = x1+x1*x2
    y = add_op(x1, mul_op(x1, x2))

    grad_x1, grad_x2 = gradient(y, [x1, x2])

    x1_val = np.array([1, 1, 1])
    x2_val = np.array([3, 2, 2])
    y_val, x1_grad_val, x2_grad_val = Executor([y, grad_x1, grad_x2]).run({x1: x1_val, x2: x2_val})
    # print(y_val,x1_grad_val,x2_grad_val)
    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val + x1_val * x2_val)
    assert np.array_equal(x1_grad_val, 1 + x2_val)
    assert np.array_equal(x2_grad_val, x1_val)


def test_add_mul_mix_2():
    x1 = Variable('x1')
    x2 = Variable('x2')
    x3 = Variable('x3')
    # y = x1*x2 + x2*x3
    y = add_op(mul_op(x1, x2), mul_op(x2, x3))

    grad_x1, grad_x2, grad_x3 = gradient(y, [x1, x2, x3])

    x1_val = 1 * np.array([1, 1, 1])
    x2_val = 2 * np.array([1, 1, 1])
    x3_val = 3 * np.array([1, 1, 1])
    y_val, x1_grad_val, x2_grad_val, x3_grad_val = Executor([y, grad_x1, grad_x2, grad_x3]).run(
        {x1: x1_val, x2: x2_val, x3: x3_val})
    # print(y_val,x1_grad_val,x2_grad_val,x3_grad_val)
    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val * x2_val + x2_val * x3_val)
    assert np.array_equal(x1_grad_val, x2_val)
    assert np.array_equal(x2_grad_val, x1_val + x3_val)
    assert np.array_equal(x3_grad_val, x2_val)


def test_add_mul_mix_3():
    x1 = Variable('x1')
    x2 = Variable('x2')
    x3 = Variable('x3')
    # y = x1+x1*x2*x3
    y = add_op(x1, mul_op(x1, mul_op(x2, x3)))

    grad_x1, grad_x2, grad_x3 = gradient(y, [x1, x2, x3])

    x1_val = 1 * np.array([1, 1, 1])
    x2_val = 2 * np.array([1, 1, 1])
    x3_val = 2 * np.array([1, 1, 1])
    y_val, x1_grad_val, x2_grad_val, x3_grad_val = Executor([y, grad_x1, grad_x2, grad_x3]).run(
        {x1: x1_val, x2: x2_val, x3: x3_val})
    # print(y_val,x1_grad_val,x2_grad_val,x3_grad_val)
    assert isinstance(y, Node)
    assert np.array_equal(y_val, x1_val + x1_val * x2_val * x3_val)
    assert np.array_equal(x1_grad_val, 1 + x2_val * x3_val)
    assert np.array_equal(x2_grad_val, x1_val * x3_val)
    assert np.array_equal(x3_grad_val, x1_val * x2_val)


def test_matmul_two_vars():
    x1 = Variable('x1')
    x2 = Variable('x2')
    y = matmul_op(x1, x2)

    grad_x1, grad_x2 = gradient(y, [x1, x2])
    x1_val = np.array([[1, 2], [3, 4]])
    x2_val = np.array([[5, 6], [7, 8]])

    y_val, x1_grad_val, x2_grad_val = Executor([y, grad_x1, grad_x2]).run({x1: x1_val, x2: x2_val})
    # print(y_val, x1_grad_val, x2_grad_val,sep='\n')
    assert isinstance(y, Node)
    assert np.array_equal(y_val, np.matmul(x1_val, x2_val))
    assert np.array_equal(x1_grad_val, np.matmul(np.ones_like(y_val), x2_val.T))
    assert np.array_equal(x2_grad_val, np.matmul(x1_val.T, np.ones_like(y_val)))


def test_mul_dep():
    x1 = Variable('x1')
    x2 = Variable('x2')
    x3 = matmul_op(x1, x2)
    x4 = mul_op(x3, x2)
    y = add_op(x4, x1)

    grad_x1, grad_x2, grad_x3, grad_x4 = gradient(y, [x1, x2, x3, x4])
    x1_val = np.array([[1, 2], [3, 4]])
    x2_val = np.array([[5, 6], [7, 8]])

    y_val = Executor([y]).run({x1: x1_val, x2: x2_val})[0]
    assert isinstance(y, Node)
    assert np.array_equal(y_val, np.matmul(x1_val, x2_val) * x2_val + x1_val)
    # res = Executor([y, grad_x1,grad_x2,grad_x3,grad_x4]).run({x1:x1_val,x2:x2_val})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='0', help='choose you case')
    args = parser.parse_args()

    print(f'===========\033[92mYou test case option is \033[91m{args.case}\033[0m==========')
    test_funcs = [test_forward, test_gradient, test_var_add_var, test_var_add_const, test_var_mul_var,
                  test_var_mul_const, test_add_mul_mix_1, test_add_mul_mix_2, test_add_mul_mix_3, test_matmul_two_vars]
    test_funcs.append(test_mul_dep)
    if args.case == 'all':
        for i in range(len(test_funcs)):
            test_funcs[i]()
            print(f'\033[94mTest case {i} passed.\033[0m')
    else:
        case_num_to_test_func = {str(i): test_funcs[i] for i in range(len(test_funcs))}
        case_num_to_test_func[args.case]()
    print('==========\033[92m ALL case passed!\033[0m=================')

    # test_one()
    # test_var_add_var() 
    # test_var_add_const()
    # test_var_mul_var()
    # test_var_mul_const()
    # test_add_mul_mix_1()
    # test_add_mul_mix_2()
    # test_add_mul_mix_3()
    # test_matmul_two_vars()
