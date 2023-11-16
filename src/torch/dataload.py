def load_data(path="biaxial_three_different_holes.xlsx", validation=False,
          split=True, extended_data=False):

    if extended_data:

        files = os.listdir("data")[:5]
        # print(files)
        df_list = []
        for file in files:
            df_list.append(pd.read_excel(os.path.join('data', file)))

        df = pd.concat(df_list)

    else:

        file = os.path.join('data', path)
        df = pd.read_excel(file)

    # n = df.shape[0]
    # print("DataFrame shape =", n)

    X = df.drop(columns=['d(psi)/d(I1)', 'd(psi)/d(I2)'])
    y = df.drop(columns=['d(psi)/d(I1)', 'I1', 'I2'])

    # print("X shape = ", X.shape)
    # print("y shape = ", y.shape)
    # print(X["I1"].values)


    # y = df.drop(columns=['I1', 'I2'])

    if not split:
        return X, y

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y,
                         test_size=0.2, random_state=2)

    if not validation:
        return X_train, X_test, y_train, y_test

    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train,
                         test_size=0.25, random_state=2)

    print("train")
    print("X shape = ", X_train.shape)
    print("y shape = ", y_train.shape)
    print("Test")
    print("X shape = ", X_test.shape)
    print("y shape = ", y_test.shape)
    print("Validation")
    print("X shape = ", X_val.shape)
    print("y shape = ", y_val.shape)

    return X_train, X_test, X_val, y_train, y_test, y_val