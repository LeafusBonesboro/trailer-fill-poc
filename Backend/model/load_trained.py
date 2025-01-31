import pickle

def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

if __name__ == "__main__":
    model = load_model("../model/trailer_fill_model_53ft.pkl")

    print("Model loaded successfully!")
