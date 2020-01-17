import pickle

def main():
    f = open('results/debug_early_stopping', 'rb')
    dataset_to_budget_to_results = pickle.load(f)
    f.close()
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
