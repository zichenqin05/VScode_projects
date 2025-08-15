import kmean_lunch as km
import sql

def 


def main(user_id):
    tag = km.lunch_recommendation(user_id)
    return tag



if __name__ == "__main__":
    user_id = int(input("请输入用户id："))
    tag = main(user_id)
    print(tag)
