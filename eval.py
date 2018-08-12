import tensorflow as tf
import numpy as np

# Generate Final Test and Validation Likelihoods
keep_prob = 1
training_loss_ave = []
validation_loss_ave = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # try to restore the pre_trained
    if restore_model_name is not None:
        print("Load the model from: {}".format(save_model_name))
        saver.restore(sess, Load_Directory + '/{}'.format(save_model_name))

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

    for p in range(10):
        _, batch_input_state_test = multi_training.getPieceBatch(
            training_pieces, batch_size,
            num_timesteps)  # not using their 'convolution' filter
        batch_input_state_test = np.array(batch_input_state_test)
        batch_input_state_test = np.swapaxes(batch_input_state_test, axis1=1,
                                             axis2=2)

        # Run Session
        feed_dict = {Note_State_Batch: batch_input_state_test,
                     timewise_state: timewise_state_val,
                     notewise_state: notewise_state_val, time_init: 0,
                     output_keep_prob: 1}
        loss_run, log_likelihood_run, note_gen_out_run = sess.run(
            [loss, log_likelihood, note_gen_out], feed_dict=feed_dict)
        training_loss_ave.append(-78 * loss_run)

        _, batch_input_state_valid = multi_training.getPieceBatch(
            validation_pieces, batch_size,
            num_timesteps)  # not using their 'convolution' filter
        batch_input_state_valid = np.array(batch_input_state_valid)
        batch_input_state_valid = np.swapaxes(batch_input_state_valid, axis1=1,
                                              axis2=2)

        # Run Session
        feed_dict = {Note_State_Batch: batch_input_state_valid,
                     timewise_state: timewise_state_val,
                     notewise_state: notewise_state_val, time_init: 0,
                     output_keep_prob: 1}
        loss_run, log_likelihood_run, note_gen_out_run = sess.run(
            [loss, log_likelihood, note_gen_out], feed_dict=feed_dict)
        validation_loss_ave.append(-78 * loss_run)

        print(p)

    # Plot the final loss
plt.plot(training_loss_ave, label="Training Log likelihood")
plt.plot(validation_loss_ave, label="Validation Log likelihood")
plt.legend()
plt.show