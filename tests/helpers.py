"""Small synthetic receipt fixtures shared by the essential training tests."""

def receipt(menu):
    return {"image_file": "example.jpg", "width": 20, "height": 30,
            "ground_truth": {"gt_parse": {"menu": menu, "total": {"total_price": "999,999"}}}}


def item(name="Meal", qty="x 2", rate="10,000", amount="20,000"):
    return {"nm": name, "cnt": qty, "unitprice": rate, "price": amount}
