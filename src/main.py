from models.registry import get_model

if __name__ == "__main__":
    demo = get_model("DemoModel", x=1, y=2)
    print(demo)
