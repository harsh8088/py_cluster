import numpy
from flask import Flask, jsonify, abort, make_response, request
# from pip._vendor.requests.auth import HTTPBasicAuth
from sklearn.metrics import pairwise_distances

import cluster_map as cluster
from kmed.constrained_kmedoids import KMedoids

app = Flask(__name__)
# auth = HTTPBasicAuth()

tasks = [
    {
        'id': 1,
        'title': "Buy",
        'description': "Cake, Cheese, Pizza, Fruit, Tylenol",
        'done': False
    },
    {
        'id': 2,
        'title': "Title",
        'description': "Need to find a good taste",
        'done': False
    },
    {
        'id': 3,
        'title': "Items",
        'description': "Random Values",
        'done': True
    }
]


@app.route('/api/v1.0/tasks', methods=['GET'])
def get_tasks():
    return jsonify({'tasks': tasks})


@app.route('/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})


# @auth.error_handler
# def unauthorized():
#     return make_response(jsonify({'error': 'Unauthorized access'}), 403)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/api/v1.0/routes', methods=['POST'])
def create_routes():
    if not request.json or not 'values' or not 'groups' in request.json:
        abort(400)
    n_list = request.json.get('values')
    n_groups = request.json.get('groups')
    nd_array = numpy.array([[0, 0]])
    x = numpy.array([[], []])

    for item in n_list:
        x = numpy.array([[item['x'], item['y']]])
        nd_array = numpy.append(nd_array, x, axis=0)
    print(nd_array)
    nd_array = numpy.delete(nd_array, 0, 0)
    val = cluster.map_clusters(nd_array, n_groups)
    n_dict = {}
    for i in range(len(val)):
        n_dict[str(i)] = val[i].tolist()

    return jsonify({'result': n_dict, 'groups': len(val)}), 201


@app.route('/api/v1.0/routes_balanced', methods=['POST'])
def create_balanced_routes():
    if not request.json or not 'values' or not 'groups' in request.json:
        abort(400)
    n_list = request.json.get('values')
    n_groups = request.json.get('groups')
    nd_array = numpy.array([[0, 0]])
    x = numpy.array([[], []])

    for item in n_list:
        x = numpy.array([[item['x'], item['y']]])
        nd_array = numpy.append(nd_array, x, axis=0)
    print(nd_array)
    nd_array = numpy.delete(nd_array, 0, 0)

    # compute distance matrix
    dist = pairwise_distances(nd_array, metric='euclidean')

    # k-medoids algorithm
    km = KMedoids(distance_matrix=dist, n_clusters=n_groups)
    km.run(max_iterations=10, tolerance=0.001)

    print(km.clusters)
    # dictionary for response
    n_dict = {}
    for k, v in km.clusters.items():
        n_dict[str(k)] = list(v)
    return jsonify({'result': n_dict, 'groups': len(n_dict.keys())}), 201


@app.route('/api/v1.0/cluster', methods=['POST'])
def make_clusters():
    if not request.json or not 'title' in request.json:
        abort(400)

    task = {
        'id': tasks[-1]['id'] + 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': task}), 201


@app.route('/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    # if 'title' in request.json and not isinstance(type(request.json['title']),str):
    #     abort(400)
    # if 'description' in request.json and not isinstance(type(request.json['description']),str) :
    #     abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description', task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify({'task': task[0]})


@app.route('/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify({'result': True})


if __name__ == '__main__':
    app.run(debug=True)
