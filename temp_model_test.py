from app import load_data, train_model

df = load_data('spam.csv')
print('rows', len(df), 'label counts', df['label'].value_counts().to_dict())
model, vec, train_df, test_df, y_test, preds, acc = train_model(df)
print('train', len(train_df), 'test', len(test_df), 'acc', acc)
print('y_test', y_test.tolist())
