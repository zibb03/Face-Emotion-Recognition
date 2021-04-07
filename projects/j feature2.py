import json

users = {
        'name' : "김헝그리",
        'age' : 13,
        'address' : {
            'city' : "서울",
            'state' : "Y"
        },
        'isAlive' : False
    }

with open('users.json', 'w') as f:
    json.dump(users, f)

    data = open('users.json', 'r').read()
    data = json.loads(data)
    print(data['name'])
