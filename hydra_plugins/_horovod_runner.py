import sys
import cloudpickle
import horovod.torch as hvd

if __name__ == '__main__':
    # load the arguments
    _, pickled_job_path = sys.argv

    # load the function that we pickled
    with open(pickled_job_path, 'rb') as pickle_file:
        (job_fn, job_result_path) = cloudpickle.load(pickle_file)

    # Initialize Horovod & set
    hvd.init()

    # Run the task!
    job_return = job_fn()

    # save the result
    with open(job_result_path, 'wb') as pickle_file:
        cloudpickle.dump(job_return, pickle_file)
