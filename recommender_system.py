import pickle
from surprise import SVD

def load_model(path: str) -> SVD:
    with open(path, 'rb') as file:  
        model = pickle.load(file)
    return model

def obtain_all_predictions(svd: SVD, df) -> list:
    predictions = []
    benefits = df.columns.to_list()
    employees = df.index.to_list()
    for uid in employees:
        for iid in benefits:
            pred = svd.predict(uid, iid)
            predictions.append((pred[0], pred[1], pred[2], pred[3]))
    
    return predictions

def get_top_n_benefits(predictions, n):
    top_n = {}
    for uid,iid, _, est in predictions:
        if uid not in top_n.keys():
            top_n[uid] = []
        top_n[uid].append((iid, est))

    for uid, user_score in top_n.items():
        user_score.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = [iid for (iid, _) in user_score[:n]]
    
    return top_n
