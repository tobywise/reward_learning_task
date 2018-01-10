# -*- coding: utf-8 -*-

from __future__ import division
from psychopy import core, visual, event, gui, monitors, data
from random import randint
import pandas as pd
import os, csv
import re
from psychopy import parallel
import pyglet
from numpy import ceil


class port:

    """
    Doesn't do anything, just makes it easy to test without parallel ports
    """

    @staticmethod
    def setData(a):
        a = 2

    @staticmethod
    def setPin(a, b):
        a = 2

port = parallel.ParallelPort(address=0x2cf8)  # TODO UNCOMMENT THIS IF YOU WANT PARALLEL PORTS TO WORK!!!!!

script_location = os.path.dirname(__file__)

# Data containers for each trial
dataCategories = ['id', 'trial_number', 'Rewarded', 'Response', 'A_reward', 'RT', 'Duration', 'Reward_prob', 'Confidence', 'Condition', 'Contingencies']
trial = dict(zip(dataCategories,['' for i in dataCategories]))

# Set monitor variables
myMon = monitors.Monitor('testMonitor')

# Intro dialogue
dialogue = gui.Dlg()
dialogue.addField('subjectID')
dialogue.show()
if dialogue.OK:
    if dialogue.data[0].isdigit(): subjectID = dialogue.data[0]
    else:
        print 'SUBJECT SHOULD BE DIGIT'
        core.quit()
else: core.quit()

if ceil(int(re.search('\d+', subjectID).group()) / 2.) % 2:  # CONTINGENCY COUNTERBALANCING
    contingencies = 'A'
else:
    contingencies = 'B'

# Make folder for dataa
saveFolder = 'data'
if not os.path.isdir(saveFolder): os.makedirs(saveFolder)

# A clock
trialClock = core.Clock()


#-------------- STIMULI ----------------
#---------------------------------------
win = visual.Window(monitor=myMon, size=(1280, 720), fullscr=True, allowGUI=False, color='black', units='deg')
fixation = visual.TextStim(win=win, height=0.8, color='white', text="+")
mainText = visual.TextStim(win=win, height=0.8, color='white', alignVert='center', alignHoriz='center', wrapWidth=30)
otherText = visual.TextStim(win=win, height=0.8, color='white', pos=(0, -2), wrapWidth=30)

image_A = visual.ImageStim(win, image=os.path.join(script_location, 'fractal3.jpg'), size=(5,5), pos=(-4, 0))
image_B = visual.ImageStim(win, image=os.path.join(script_location, 'fractal4.jpg'), size=(5,5), pos=(4, 0))
background_A = visual.Rect(win, width=6, height=6, fillColor='#989898', lineColor='gray', pos=(-4, 0))
background_B = visual.Rect(win, width=6, height=6, fillColor='#989898', lineColor='gray', pos=(4, 0))

circle = visual.ImageStim(win, image=os.path.join(script_location, 'pound.png'), size=4)
loss = visual.ImageStim(win, image=os.path.join(script_location, 'loss.png'), size=4)

win_text = visual.TextStim(win=win, height=2, color='#7FFF00', pos=(0, -3.5), wrapWidth=30, bold=True, text='+20')
loss_text = visual.TextStim(win=win, height=2, color='red', pos=(0, -3.5), wrapWidth=30, bold=True, text='-20')

rectangle = visual.Rect(win, width=5, height=5, fillColor='#989898', lineColor='gray')

select_rect = visual.Rect(win, width=6, height=6, pos=(-3, 0), fillColor=None, lineColor='yellow', lineWidth=4)

inst_text = visual.TextStim(win=win, height=0.7, color='white', pos=(0, -7), wrapWidth=30, bold=True)
inst_text2 = visual.TextStim(win=win, height=0.7, color='white', pos=(0, -7), wrapWidth=30, bold=True)
inst_text3 = visual.TextStim(win=win, height=0.7, color='white', pos=(0, -7), wrapWidth=30, bold=True)
inst_text4 = visual.TextStim(win=win, height=0.7, color='white', pos=(0, -7), wrapWidth=30, bold=True)

confidence_box = visual.Rect(win=win, width=15, height=2, pos=(0, 0), lineColor='white', lineWidth=3)
confidence_line = visual.Rect(win=win, width=0.2, height=2, pos=(0, 0), lineColor=None, lineWidth=3, fillColor='white')
marker_1 = visual.Rect(win=win, width=0.1, height=2, pos=(-3.75, 0), lineColor=None, lineWidth=3, fillColor='grey')
marker_2 = visual.Rect(win=win, width=0.1, height=2, pos=(0, 0), lineColor=None, lineWidth=3, fillColor='grey')
marker_3 = visual.Rect(win=win, width=0.1, height=2, pos=(3.75, 0), lineColor=None, lineWidth=3, fillColor='grey')

guess_text = visual.TextStim(win=win, text="Guess", pos=(-7.5, 2), color='white')
confident_text = visual.TextStim(win=win, text="Confident", pos=(7.5, 2), color='white')

# keys
ansKeys = ['left', 'right']
quitKeys = ['q','esc','escape']

#keyboard for ratings

key=pyglet.window.key
keyboard = key.KeyStateHandler()
win.winHandle.push_handlers(keyboard)

"""
FUNCTIONS
"""


def showText(text, show_fixation=True):
    continueTrial = True
    show_text = True

    mainText.setText(text)

    trialClock.reset()

    while continueTrial:
        port.setData(0) #TODO reset pins here?

        response = event.getKeys(keyList=['space'] + quitKeys, timeStamped=trialClock)

        if response and response[0][0] == 'space':
            show_text = False
        elif response and response[0][0] in quitKeys:
            core.quit()

        if show_text:
            mainText.draw()

        else:
            if show_fixation:
                fixation.draw()
                win.flip()
                core.wait(2)
            event.clearEvents()
            continueTrial = False

        win.flip()


def confidence_rating():

    if keyboard[key.L] and confidence_line.pos[0] < 7.4:
        confidence_line.pos += (0.2, 0)

    elif keyboard[key.A] and confidence_line.pos[0] > -7.4:
        confidence_line.pos -= (0.2, 0)

    marker_1.draw()
    marker_2.draw()
    marker_3.draw()

    confidence_box.draw()
    confidence_line.draw()
    guess_text.draw()
    confident_text.draw()

    return (confidence_line.pos[0] + 7.4) / 15.


def runBlock(trial_info, training=False, structured=False, n_trials=None, response_keys=('a', 'l')):

    trial_info = pd.read_csv(trial_info)

    trial_number = trial_info.trial_number.tolist()
    trial_type = trial_info.trial_type.tolist()
    a_reward = trial_info.A_reward.tolist()
    reward_prob = trial_info.Reward_probability.tolist()
    confidence_trial = trial_info.Confidence_trial.tolist()

    if int(re.search('\d+', subjectID).group()) % 2:   # REWARD IMAGE COUNTERBALANCING
        condition = 'A'
        a_reward = [1 - a for a in a_reward]
        reward_prob = [1 - r for r in reward_prob]
    else:
        condition = 'B'

    correct_responses = 0

    expInfo = {}
    expInfo['date'] = data.getDateStr()

    # Set up .csv save function
    if not training:
        saveFile = saveFolder+'/learning_data_' +str(subjectID)+ '_' + condition + '_' + expInfo['date'] + '.csv'      # Filename for save-data
        csvWriter = csv.writer(open(saveFile, 'wb'), delimiter=',').writerow        # The writer function to csv
        csvWriter(dataCategories)                                                   # Writes title-row in csv

    # DURATIONS

    iti = 1
    stim_duration = 1.5
    fix_duration = 0.5
    reward_duration = 0.8

    # Loop through trials
    if not n_trials:
        n_trials = len(trial_number)

    for i in range(n_trials):

        event.clearEvents()
        trialClock.reset()

        continueTrial = True
        continueConfidence = True
        response_made = False
        confidence = 'NA'

        if confidence_trial[i]:
            confidence_duration = 3
        else:
            confidence_duration = 0

        confidence_line.setPos((0, 0))

        # iti = 1.2 + (randint(-250, 250)) / 1000.

        rewarded = False

        pport_set = False

        response = None

        t = 0

        while continueTrial:
            if not training:
                port.setData(0) #TODO or here?

            if not structured:
                t = trialClock.getTime() # What's the time Mr Wolf?

            ### BREAK TRIAL

            if trial_type[i] == 'break':

                if t < 10:

                    mainText.setText('Break')
                    otherText.setText('Starting again in {0}'.format(10-int(t)))
                    mainText.draw()
                    otherText.draw()
                    port.setData(200) #TODO


                elif 10 <= t < 12:
                    fixation.draw()
                else:
                    continueTrial = False

            elif trial_type[i] == 'end':

                if t < 20:
                    mainText.setText('End of experiment, thank you!')
                    winnings = str((correct_responses * 2) / 100.)
                    if len(winnings) < 4:
                        winnings = winnings + str(0)
                    otherText.setText('You won {0}'.format(winnings))
                    mainText.draw()
                    otherText.draw()

                else:
                    continueTrial = False

            #### NORMAL TRIAL

            else:

                if 0 <= t < stim_duration:

                    background_A.draw()
                    background_B.draw()
                    image_A.draw()
                    image_B.draw()
                    if not training:
                        port.setPin(2,1) #TODO


                    if structured:
                        inst_text2.draw()

                    if not response:
                        response = event.getKeys(keyList=list(response_keys) + quitKeys, timeStamped=trialClock)

                    else:
                        if response[-1][0] in quitKeys:
                            core.quit()
                        response_made = True
                        trial['RT'] = response[0][1]
                        trial['Response'] = response[0][0]
                        if response[0][0] == response_keys[0]:
                            select_rect.setPos((-4, 0))
                        elif response[0][0] == response_keys[1]:
                            select_rect.setPos((4, 0))

                        if a_reward[i] and response[0][0] == response_keys[0]:
                            rewarded = True
                            trial['Rewarded'] = 1
                            correct_responses += 1
                            if not training:
                                port.setPin(3,1)# TODO marker here for response
                        elif not a_reward[i] and response[0][0] == response_keys[1]:
                            rewarded = True
                            trial['Rewarded'] = 1
                            correct_responses += 1
                            if not training:
                                port.setPin(3,1)# TODO marker here for response
                        else:
                            trial['Rewarded'] = 0
                            if not training:
                                port.setPin(4,1)# TODO marker here for incorrect response

                        select_rect.draw()

                        if structured:
                            win.flip()
                            core.wait(3)
                            if confidence_trial[i]:
                                t = stim_duration + fix_duration + 0.1
                            else:
                                t = stim_duration + fix_duration + confidence_duration  + fix_duration + 0.1
                            event.clearEvents()


                if stim_duration <= t < stim_duration +  fix_duration and not structured:
                    fixation.draw()

                if stim_duration + fix_duration <= t < stim_duration + fix_duration + confidence_duration:  # CONFIDENCE RATING

                    # TODO set port

                    if structured:
                        inst_text3.draw()

                    if response:
                        port.setPin(8, 1)
                        confidence = confidence_rating()
                    else:
                        fixation.draw()

                    if structured:
                        t = stim_duration + fix_duration + 0.1
                        space_pressed = event.getKeys()
                        if len(space_pressed) > 0:
                            if space_pressed[0] == 'space':
                                event.clearEvents()
                                t = stim_duration + fix_duration + confidence_duration  + fix_duration + 0.1


                if stim_duration + fix_duration + confidence_duration <= t < stim_duration + fix_duration + confidence_duration + fix_duration and not structured:
                    fixation.draw()

                if stim_duration + fix_duration + confidence_duration  + fix_duration <= t < stim_duration + fix_duration + confidence_duration  + fix_duration + reward_duration:

                    if structured:
                        inst_text4.draw()

                    if response or structured:
                        rectangle.draw()
                        if rewarded:
                            circle.draw()
                            win_text.draw()
                            if not training:
                                port.setPin(5, 1)# TODO marker here for reward
                        else:
                            loss.draw()
                            loss_text.draw()
                            if not training:
                                port.setPin(6, 1)# TODO marker here for loss
                    else:
                        fixation.draw()

                    if structured:
                        response = event.getKeys(keyList=['space'] + quitKeys, timeStamped=trialClock)

                        if response and response[0][0] == 'space':
                            continueTrial = False
                            event.clearEvents()


                if stim_duration + fix_duration + confidence_duration + fix_duration + reward_duration <= t < stim_duration + fix_duration + confidence_duration + fix_duration+  reward_duration + iti:
                    fixation.draw()
                    # parallel.setData(0)

                if stim_duration + fix_duration + confidence_duration + fix_duration + reward_duration + iti <= t:
                    fixation.draw()
                    continueTrial = False
                
            win.flip()

            trial['id'] = subjectID
            trial['trial_number'] = i
            trial['Duration'] = str(t)
            trial['Reward_prob'] = reward_prob[i]
            trial['A_reward'] = a_reward[i]
            trial['Confidence'] = confidence
            trial['Condition'] = condition
            trial['Contingencies'] = contingencies


            if not continueTrial:
                event.clearEvents()
                if not response_made:
                    trial['RT'] = 'Invalid'
                    trial['Response'] = 'Invalid'
                trialClock.reset()
                if not training:
                    csvWriter([trial[category] for category in dataCategories])
                break

            # check for quit (the [Esc] key)
            if event.getKeys(["escape"]):
                core.quit()

"""
Run the experiment
"""

instruction_text = ["In this task you will have the opportunity to win some money\n\n"
                    "You will see two pictures on each trial and you will have to select one of them\n\n"
                    "One of the cards will win you money, the other will lose money\n\n"
                    "If you choose the rewarded picture, the money will be added to your total winnings. "
                    "If you choose the incorrect picture, it will be deducted\n\n"
                    "A proportion of the your total winnings will be added to your payment at the end of the experiment, so "
                    "try to win as much as possible and lose as little as possible!"
                    "\n\nPress the space bar to continue",

                    "It won't be clear at first which picture gives a win and which gives a loss, so you'll have to learn this during the task\n\n"
                    "Importantly, the picture that has the highest chance of giving you a reward will sometimes change\n\n"
                    "It won't always be obvious which picture is giving you a reward\n\n"
                    "This means that it might be difficult at times, but try your best"
                    "\n\nPress the space bar to continue,",

                    "You will also be asked on some trials to rate how confident you are in your response on a scale\n\n"
                    "This can range from a complete guess to being completely certain\n\n"

                    "Before we begin, we'll go through the experiment at your own pace so you can learn how the trials work\n\n"
                    "Press the space bar to continue"]

practice_text = "Now try some practice trials\n\n" \
                "These will run at their own pace, so you don't need to press the space bar once you've begun"


trial_info = os.path.join(script_location, 'contingencies_5{0}.csv'.format(contingencies))
practice_info = os.path.join(script_location, 'contingencies_training.csv')

response_keys = ['a', 'l']  # Don't change this or everything will break

# STRUCTURED PRACTICE INSTRUCTIONS

instructions = ["One of these images is more likely to win you money, and one is more likely to lose you money - your task is to learn which is the best one\n"
                                          "If you choose the winning picture, you will win a small amount of money, if not you'll lose a small amount\n"
                                          "It won't always be clear which one is the winning option, so sometimes you'll feel like you're guessing\n"
                                          "You can choose which image you think will win using the {0} or {1} keys: {0} = left, {1} = right\n"
                                          "If you don't respond quickly enough, you won't see whether you've won the reward or not\n\n"
                                          "Choose one of the pictures to continue".format(response_keys[0].capitalize(), response_keys[1].capitalize()),
                "On some trials you will be asked to rate how confident you are in your choice after making it\n\n"
                                          "You can move the line left or right using the A and L keys\n\n"
                                          "Left of the box = guess, right of the box = completely confident in your choice\n\n"
                                          "Make sure you use the whole range of the scale when making your ratings. "
                                          "If you're guessing, move the slider to the left, if you're sure, move it to the right\n\n"
                                          "You have 2 seconds to make the choice. "
                                          "Try to rate your confidence as accurately as you can, but it doesn't matter if it's not perfectly accurate\n\n"
                                          "Try moving the rating, and then press the space bar to continue".format(response_keys[0].capitalize(), response_keys[1].capitalize()),
                "This screen shows whether you won or lost\n"
                                          "If you chose the rewarded image you'll see a coin\n"
                                          "If you didn't, you'll see a coin with a cross through it\n\n"
                                          "Press space to continue"                          
                                          ]
                                          
inst_text2.setText(instructions[0])
inst_text3.setText(instructions[1])
inst_text4.setText(instructions[2])

# INSTRUCTIONS

for i in instruction_text:
    showText(i, show_fixation=False)

# STRUCTURED PRACTICE

runBlock(practice_info, training=True, structured=True, n_trials=3)

# PRACTICE

showText(practice_text, show_fixation=True)
runBlock(practice_info, training=True)

# EXPERIMENT

# TODO how to get pp markers to only be sent on real task (not practice), and how to add start of block code
# TODO hi Laura

# parallel.setData(200)  #send start of block code
# core.wait(0.2)
# parallel.setData(0)
# core.wait(0.2)
showText("Now we're ready to start the task, press space to start!", show_fixation=True)
runBlock(trial_info, training=False)
showText("The task has finished, thank you for taking part!", show_fixation=False)

core.quit()

