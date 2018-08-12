import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
import time
import os
import random
import midi_musical_matrix
import data
import multi_training
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
from MyFunctions import Input_Kernel, LSTM_TimeWise_Training_Layer, LSTM_NoteWise_Layer, Loss_Function

# Import All Training and Validation Data
# Convert Entire Music .MIDI set to list of musical 'pieces'
# During training runs, getPieceBatch will return a tensor for Note_State_Batch, and corresponding Note_State_Expand
# Note_State_Expand will be fed into the graph input, and Note_State_Batch will be used for the loss function.


def get_feed_dict(time_tensors,note_tensors,batch_input,timewise_state_val,notewise_state_val,t,keep_prob):
    feed_dict = {'node_state_batch:0': batch_input,
                 time_tensors: timewise_state_val,
                 note_tensors: notewise_state_val, "time_init:0": t,
                 "output_keep_prob:0": keep_prob}
    return feed_dict

# parameters
def train():
    Working_Directory = os.getcwd()
    Music_Directory = "../proyecto-beatles/midis_por_instrumento/"
    print(Working_Directory)
    Midi_Directories = ["Reed_Voice"]
    max_time_steps = 256  # only files atleast this many 16th note steps are saved
    num_validation_pieces = 1
    practice_batch_size = 15
    practice_num_timesteps = 128
    num_t_units=[200, 200]

    start_time = time.time()
    max_iteration = 10000 #50000
    loss_hist = []
    loss_valid_hist = []
    restore_model_name = None #'Long_Train'
    save_model_name = 'Long_Train_256'
    batch_size = 5
    num_timesteps = 256
    keep_prob = .5
    num_n_units = [100, 100]



    # Gather the training pieces from the specified directories
    training_pieces = {}
    limit_train = 2
    for f in range(len(Midi_Directories)):
        Training_Midi_Folder = Music_Directory + Midi_Directories[f]

        training_pieces = {**training_pieces,**multi_training.loadPieces(Training_Midi_Folder,max_time_steps,max_elements=limit_train)}

    # Set aside a random set of pieces for validation purposes
    validation_pieces = {}
    for v in range(num_validation_pieces):
        index = random.choice(list(training_pieces.keys()))
        validation_pieces[index] = training_pieces.pop(index)

    print('')
    print('Number of training pieces = ', len(training_pieces))
    print('Number of validation pieces = ', len(validation_pieces))

    # Generate sample Note State Matrix for dimension measurement and numerical checking purposes
    # (Using external code to generate the Note State Matrix but using our own NoteInputForm (as defined in author's code) function


    _, sample_state = multi_training.getPieceBatch(training_pieces,
                                                   practice_batch_size,
                                                   practice_num_timesteps)
    sample_state = np.array(sample_state)
    sample_state = np.swapaxes(sample_state, axis1=1, axis2=2)
    print('Sample of State Input Batch: shape = ', sample_state.shape)



    # Build the Model Graph:
    tf.reset_default_graph()
    print('Building Graph...')
    #Capture number of notes from sample
    num_notes = sample_state.shape[1]

    # Graph Input Placeholders
    Note_State_Batch = tf.placeholder(dtype=tf.float32, shape=[None, num_notes, None, 2],name='node_state_batch')
    time_init = tf.placeholder(dtype=tf.int32, shape=(),name='time_init')

    #Generate expanded tensor from batch of note state matrices
    # Essential the CNN 'window' of this network
    Note_State_Expand = Input_Kernel(Note_State_Batch, Midi_low=24, Midi_high=101, time_init=time_init)

    print('Note_State_Expand shape = ', Note_State_Expand.get_shape())


    # lSTM Time Wise Training Graph

    output_keep_prob = tf.placeholder(dtype=tf.float32, shape=(),name="output_keep_prob")

    # Generate initial state (at t=0) placeholder
    timewise_state=[]
    for i in range(len(num_t_units)):
        timewise_c=tf.placeholder(dtype=tf.float32, shape=[None, num_t_units[i]],name='timewise_c{0}'.format(i)) #None = batch_size * num_notes
        timewise_h=tf.placeholder(dtype=tf.float32, shape=[None, num_t_units[i]],name='timewise_h{0}'.format(i))
        timewise_state.append(LSTMStateTuple(timewise_h, timewise_c))

    timewise_state=tuple(timewise_state)


    timewise_out, timewise_state_out = LSTM_TimeWise_Training_Layer(input_data=Note_State_Expand, state_init=timewise_state, output_keep_prob=output_keep_prob)


    print('Time-wise output shape = ', timewise_out.get_shape())




    #LSTM Note Wise Graph



    # Generate initial state (at n=0) placeholder
    notewise_state=[]
    for i in range(len(num_n_units)):
        notewise_c=tf.placeholder(dtype=tf.float32, shape=[None, num_n_units[i]],name='notewise_c{0}'.format(i)) #None = batch_size * num_timesteps
        notewise_h=tf.placeholder(dtype=tf.float32, shape=[None, num_n_units[i]],name='notewise_h{0}'.format(i))
        notewise_state.append(LSTMStateTuple(notewise_h, notewise_c))

    notewise_state=tuple(notewise_state)


    y_out, note_gen_out = LSTM_NoteWise_Layer(timewise_out, state_init=notewise_state, output_keep_prob=output_keep_prob)
    note_gen_out = tf.identity(note_gen_out, name="note_gen_out")

    p_out = tf.sigmoid(y_out)
    print('y_out shape = ', y_out.get_shape())
    print('generated samples shape = ', note_gen_out.get_shape())



    # Loss Function and Optimizer

    loss, log_likelihood = Loss_Function(Note_State_Batch, y_out)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate = 1).minimize(loss)
    #optimizer = tf.train.RMSPropOptimizer
    print('Graph Building Complete')

    # Training



    # Save Model
    Output_Directory = Working_Directory + "/Output/" + save_model_name
    directory = os.path.dirname(Output_Directory)

    try:
        print('creating new destination folder')
        os.mkdir(directory)
    except:
        print('destination folder exists')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # try to restore the pre_trained
        if restore_model_name is not None:
            Load_Directory = Working_Directory + "/Output/" + restore_model_name

            print("Load the model from: {}".format(restore_model_name))
            saver.restore(sess, Load_Directory + '/{}'.format(restore_model_name))

        # Initial States
        timewise_state_val = []
        for i in range(len(num_t_units)):
            c_t = np.zeros((batch_size * num_notes, num_t_units[
                i]))  # start every batch with zero state in LSTM time cells
            h_t = np.zeros((batch_size * num_notes, num_t_units[i]))
            timewise_state_val.append(LSTMStateTuple(h_t, c_t))

        notewise_state_val = []
        for i in range(len(num_n_units)):
            c_n = np.zeros((batch_size * num_timesteps, num_n_units[
                i]))  # start every batch with zero state in LSTM time cells
            h_n = np.zeros((batch_size * num_timesteps, num_n_units[i]))
            notewise_state_val.append(LSTMStateTuple(h_n, c_n))


        # state,output tensors
        note_tensors = []
        time_tensors = []
        for i in range(len(num_n_units)):
            note_tensors.append(LSTMStateTuple('notewise_h{0}:0'.format(i),
                                               'notewise_c{0}:0'.format(i)))

        for i in range(len(num_t_units)):
            time_tensors.append(LSTMStateTuple('timewise_h{0}:0'.format(i),
                                               'timewise_c{0}:0'.format(i)))
        note_tensors = tuple(note_tensors)
        time_tensors = tuple(time_tensors)




        # Training Loop
        for iteration in range(max_iteration + 1):

            # Generate random batch of training data
            if (iteration % 100 == 0):
                print('Obtaining new batch of pieces')
                _, batch_input_state = multi_training.getPieceBatch(training_pieces,
                                                                    batch_size,
                                                                    num_timesteps)  # not using their 'convolution' filter
                batch_input_state = np.array(batch_input_state)
                batch_input_state = np.swapaxes(batch_input_state, axis1=1, axis2=2)



            # Run Session
            feed_dict = get_feed_dict(time_tensors,note_tensors,batch_input_state,timewise_state_val,notewise_state_val,0,keep_prob)


            loss_run, log_likelihood_run, _, note_gen_out_run = sess.run(
                [loss, log_likelihood, optimizer, note_gen_out],
                feed_dict=feed_dict)

            # Periodically save model and loss histories
            if (iteration % 1000 == 0) & (iteration > 0):
                save_path = saver.save(sess, Output_Directory + '/{}'.format(
                    save_model_name))
                print("Model saved in file: %s" % save_path)
                np.save(Output_Directory + "/ training_loss.npy", loss_hist)
                np.save(Output_Directory + "/ valid_loss.npy", loss_valid)

            # Regularly Calculate Validation loss and store both training and validation losses
            if (iteration % 100) == 0 & (iteration > 0):
                # Calculation Validation loss
                _, batch_input_state_valid = multi_training.getPieceBatch(
                    validation_pieces, batch_size,
                    num_timesteps)  # not using their 'convolution' filter
                batch_input_state_valid = np.array(batch_input_state_valid)
                batch_input_state_valid = np.swapaxes(batch_input_state_valid,
                                                      axis1=1, axis2=2)
                feed_dict = {Note_State_Batch: batch_input_state_valid,
                             timewise_state: timewise_state_val,
                             notewise_state: notewise_state_val, time_init: 0,
                             output_keep_prob: keep_prob}
                loss_valid, log_likelihood_valid = sess.run([loss, log_likelihood],
                                                            feed_dict=feed_dict)

                print('epoch = ', iteration, ' / ', max_iteration, ':')
                print('Training loss = ', loss_run, '; Training log likelihood = ',
                      log_likelihood_run)
                print('Validation loss = ', loss_valid,
                      '; Validation log likelihood = ', log_likelihood_valid)

                loss_hist.append(loss_run)
                loss_valid_hist.append(loss_valid)

            # Periodically generate Sample of music
            if (iteration % 3000) == 0 and (iteration > 0):
                generate_midi(sess, num_notes,num_t_units, num_n_units, "midi_iteracion{0}".format(iteration),timewise_state_out)

    end_time = time.time()

    print('Training time = ', end_time - start_time, ' seconds')



    # Plot the loss histories
    plt.plot(loss_hist, label="Training Loss",style='*--')
    plt.plot(loss_valid_hist, label="Validation Loss",style='*--')
    plt.legend()
    plt.savefig(os.path.join(Output_Directory,"train-val_loss.png"))




def load_and_generate_midi(model_checkpoint_folder):
    # Music Generation

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        print("Load the model from: {}".format(model_checkpoint_folder))
        saver.restore(sess, model_checkpoint_folder)

        num_notes=78
        num_t_units = [200,200]
        num_n_units = [200,200]
        out_name = "modelo_recuperado"

        generate_midi(sess, num_notes, num_t_units, num_n_units, out_name)

def generate_midi(sess,num_notes,num_t_units,num_n_units,out_name,timewise_state_out):
    """
        Music Generation
        input = initial note vector
        for t = 1:Tsong
        input --> input kernel
        run through 1 'call' of Model LSTM with present parameters / states
        run through note-wise LSTM block as normally done to produce vector of generated samples
        input = generated samples
        music_sequence.append(input)
        store batch of music sequences in .MIDI files
    """
    T_gen = 64 * 16
    batch_gen_size = 10
    keep_prob = 1 # todo estaba en 0.5 ??

    # state,output tensors
    note_tensors = []
    time_tensors = []
    for i in range(len(num_n_units)):
        note_tensors.append(LSTMStateTuple('notewise_h{0}:0'.format(i),
                                           'notewise_c{0}:0'.format(i)))

    for i in range(len(num_t_units)):
        time_tensors.append(LSTMStateTuple('timewise_h{0}:0'.format(i),
                                           'timewise_c{0}:0'.format(i)))
    note_tensors = tuple(note_tensors)
    time_tensors = tuple(time_tensors)

    # start with initial Note_State_Batch with 't' dimension = 1 (can still a batch of samples run in parallel)
    notes_gen_initial = np.zeros((batch_gen_size, num_notes, 1, 2))

    # Initial States
    notes_gen = notes_gen_initial

    timewise_state_val = []
    for i in range(len(num_t_units)):
        c = np.zeros((batch_gen_size * num_notes, num_t_units[i]))  # start first time step with zero state in LSTM time cells
        h = np.zeros((batch_gen_size * num_notes, num_t_units[i]))
        timewise_state_val.append(LSTMStateTuple(h, c))

    notewise_state_val = []
    for i in range(len(num_n_units)):
        c = np.zeros((batch_gen_size * 1, num_n_units[i]))  # start every batch with zero state in LSTM time cells
        h = np.zeros((batch_gen_size * 1, num_n_units[i]))
        notewise_state_val.append(LSTMStateTuple(h, c))

    notes_gen_arr = []


    for t in range(T_gen):

        feed_dict = get_feed_dict(time_tensors, note_tensors, notes_gen,
                                  timewise_state_val, notewise_state_val, t,
                                  keep_prob)

        timewise_state_val, notes_gen = np.squeeze(sess.run([timewise_state_out, "note_gen_out:0"],feed_dict=feed_dict),axis=2)
        notes_gen_arr.append(np.squeeze(notes_gen))

        if t % 50 == 0:
            print('Timestep = ', t)

    # Save Generate Notes to .MIDI file
    notes_gen_out = np.stack(notes_gen_arr, axis=2)
    notes_gen_out = np.swapaxes(notes_gen_out, axis1=1, axis2=2)

    os.makedirs("out_midis",exist_ok=True)
    for iter in range(batch_gen_size):
        name_out = out_name
        out_path = os.path.join("out_midis",name_out+"_"+str(iter))
        midi_out = midi_musical_matrix.noteStateMatrixToMidi(notes_gen_out[iter, :, :, :], name=out_path)

    print('Midi files saved')

    pass

if __name__ == "__main__":
    train()