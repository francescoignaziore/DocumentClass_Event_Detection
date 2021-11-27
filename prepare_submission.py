from path import Path
import pandas as pd

path_to_test_ds = '../../03_dataset/20210312'
path_to_runs = 'test_results'
path_to_save = 'send_for_submission'

run = 't7'
filter_run = ['st1', 'st2']

for filter_r in filter_run:

    if filter_r == 'st1':
        filter_st = ['en', 'es', 'pr', 'hi']
    else:
        filter_st = ['en', 'es', 'pr']

    for filter_l in filter_st:
        print(f'Preparing results for: {filter_r} / {filter_l}...')

        file_test_db = Path(f'{path_to_test_ds}/{filter_r}').glob(f'{filter_l}-test.json')
        file_result = Path(path_to_runs).glob(f'*{filter_r}_{filter_l}/test_results*.txt')

        # Preper as 'final_submission.json' for server upload
        store = dict()
        for file in file_test_db:
            print(file)
            data_test_ds = pd.read_json(file, lines=True)

        for file in file_result:
            print(file)
            data_prediction = pd.read_csv(file, sep='\t')

        assert len(data_test_ds) == len(data_prediction)

        if filter_r == 'st2':
            store['id'] = data_test_ds['id']
            store['prediction'] = data_prediction['prediction']
        else:
            store['prediction'] = data_prediction['prediction']
            store['id'] = data_test_ds['id']

        store = pd.DataFrame(store).reset_index(drop=True)

        store.to_json(f'{path_to_save}/submission_{run}_{filter_r}_{filter_l}.json', orient='records', lines=True,
                      force_ascii=False)

print('Done')

# Check zipped data
'''
df = pd.read_csv(Path(f'{path_to_save}').glob('*.zip')[0])
df.shape
'''