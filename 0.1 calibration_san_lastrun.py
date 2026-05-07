#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on May 07, 2026, at 11:26
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from ds5_initiation
from ds5_none_uncertainty import DS5PulseConfig, fire_none_uncertainty_pulse

# Run 'Before Experiment' code from first_over_compilation
import pandas as pd

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = '0.1 calibration_san'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': 'sub-000',
    'ramp_start_ma': '1.0',
    'ramp_increment_ma': '1.0',
    'nramps ': '3',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/calibration/%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\labmp\\Desktop\\git\\replay_pain\\Pain-Bayes\\0.1 calibration_san_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('welcome_end') is None:
        # initialise welcome_end
        welcome_end = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='welcome_end',
        )
    if deviceManager.getDevice('end_instructions') is None:
        # initialise end_instructions
        end_instructions = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_instructions',
        )
    if deviceManager.getDevice('end_verification') is None:
        # initialise end_verification
        end_verification = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_verification',
        )
    if deviceManager.getDevice('confirm_continue') is None:
        # initialise confirm_continue
        confirm_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='confirm_continue',
        )
    if deviceManager.getDevice('slider_pain_resp_2') is None:
        # initialise slider_pain_resp_2
        slider_pain_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='slider_pain_resp_2',
        )
    if deviceManager.getDevice('confirm_resp') is None:
        # initialise confirm_resp
        confirm_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='confirm_resp',
        )
    if deviceManager.getDevice('continue_first_over') is None:
        # initialise continue_first_over
        continue_first_over = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='continue_first_over',
        )
    if deviceManager.getDevice('start_second_p') is None:
        # initialise start_second_p
        start_second_p = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_second_p',
        )
    if deviceManager.getDevice('admin_2') is None:
        # initialise admin_2
        admin_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='admin_2',
        )
    if deviceManager.getDevice('slider_pain_resp_3') is None:
        # initialise slider_pain_resp_3
        slider_pain_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='slider_pain_resp_3',
        )
    if deviceManager.getDevice('exit') is None:
        # initialise exit
        exit = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='exit',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome_2" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='add welcome text\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    welcome_end = keyboard.Keyboard(deviceName='welcome_end')
    # Run 'Begin Experiment' code from ds5_initiation
    from pathlib import Path
    import numpy as np
    import pandas as pd
    
    
    def as_bool(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y", "oui"}
    
    N_RAMPS = int(expInfo["nramps"])
    RAMP_START_MA = float(expInfo["ramp_start_ma"])
    RAMP_INCREMENT_MA = float(expInfo["ramp_increment_ma"])
    RAMP_MAX_MA = 25
    
    ds5_config = DS5PulseConfig(
        v_per_ma=0.4,
        pulse_width_ms=1,
        pulse_frequency_hz=100,
        train_duration_ms=500,
        record_monitor=True,
    )
    
    calib_dir = Path(_thisDir) / "data" / f"{expInfo['participant']}_{data.getDateStr(format='%Y-%m-%d')}" / "Calibration"
    waveform_dir = calib_dir / "ds5_waveforms"
    
    series_trial = []
    ramp_trial = []
    intensities_list = []
    ratings_list = []
    fired_list = []
    last_ramp_tolerated_ma = None
    ratings_random = []
    fired = 0
    
    # This replaces the old therm_intensity list.
    electrical_intensity = np.around(
        np.arange(RAMP_START_MA, RAMP_MAX_MA + RAMP_INCREMENT_MA / 2, RAMP_INCREMENT_MA),
        3,
    )
    # Run 'Begin Experiment' code from fixed_variables
    import random
    nramps = 3
    
    # --- Initialize components for Routine "instruc" ---
    explain_participant = visual.TextStim(win=win, name='explain_participant',
        text='add instructions here\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_instructions = keyboard.Keyboard(deviceName='end_instructions')
    
    # --- Initialize components for Routine "initiation_2" ---
    verification = visual.TextStim(win=win, name='verification',
        text='Experimenter verify - zeroing + active pulse ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_verification = keyboard.Keyboard(deviceName='end_verification')
    
    # --- Initialize components for Routine "stimulation_start" ---
    next_stim = visual.TextStim(win=win, name='next_stim',
        text='Appuyer sur P pour recevoir la prochaine stimulation\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    confirm_continue = keyboard.Keyboard(deviceName='confirm_continue')
    
    # --- Initialize components for Routine "jitter_cross" ---
    jitter_cross_component = visual.TextStim(win=win, name='jitter_cross_component',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "fire_electrical" ---
    cross_fire = visual.TextStim(win=win, name='cross_fire',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "rating_scale" ---
    slider_pain_2 = visual.Slider(win=win, name='slider_pain_2',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=['Aucune\ndouleur', 'Pire douleur\nimaginable'], ticks=[0, 100], granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.04,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    slider_pain_resp_2 = keyboard.Keyboard(deviceName='slider_pain_resp_2')
    main_prompt_2 = visual.TextStim(win=win, name='main_prompt_2',
        text="Veuillez évaluer l'intensité de la douleur ressentie.",
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    space_prompt_2 = visual.TextStim(win=win, name='space_prompt_2',
        text="Appuyez sur la barre d'espacement pour valider.",
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "confirm_continue_2" ---
    # Run 'Begin Experiment' code from confirm_code
    series_trial = []
    ramp_trial = []
    intensities_list = []
    ratings_list = []
    fired_list = []
    confirm_text = visual.TextStim(win=win, name='confirm_text',
        text="Pensez-vous être capable de recevoir une stimulation d'une plus grande intensité ? ",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    confirm_resp = keyboard.Keyboard(deviceName='confirm_resp')
    
    # --- Initialize components for Routine "compile_ramps" ---
    first_over = visual.TextStim(win=win, name='first_over',
        text='First part over - add text',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    continue_first_over = keyboard.Keyboard(deviceName='continue_first_over')
    
    # --- Initialize components for Routine "second_phase_start" ---
    second_instructions = visual.TextStim(win=win, name='second_instructions',
        text='add instructions seconde phase ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    start_second_p = keyboard.Keyboard(deviceName='start_second_p')
    
    # --- Initialize components for Routine "trials_part_2" ---
    # Run 'Begin Experiment' code from admin_2_start
    curr_item_scd = -1
    admnisiter_stim_2 = visual.TextStim(win=win, name='admnisiter_stim_2',
        text='P to get stim again - modify text her',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    admin_2 = keyboard.Keyboard(deviceName='admin_2')
    
    # --- Initialize components for Routine "jitter_2" ---
    jitter_fix_text_2 = visual.TextStim(win=win, name='jitter_fix_text_2',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "send_electrical_2" ---
    fix_2 = visual.TextStim(win=win, name='fix_2',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "ratings_2" ---
    slider_pain_3 = visual.Slider(win=win, name='slider_pain_3',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=['Aucune\ndouleur', 'Pire douleur\nimaginable'], ticks=[0, 100], granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.04,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    slider_pain_resp_3 = keyboard.Keyboard(deviceName='slider_pain_resp_3')
    main_prompt_3 = visual.TextStim(win=win, name='main_prompt_3',
        text="Veuillez évaluer l'intensité de la douleur ressentie.",
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    space_prompt_3 = visual.TextStim(win=win, name='space_prompt_3',
        text="Appuyez sur la barre d'espacement pour valider.",
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "thank_you" ---
    # Run 'Begin Experiment' code from end_code
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    
    
    thank_you_text = visual.TextStim(win=win, name='thank_you_text',
        text='Thank you :) ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    exit = keyboard.Keyboard(deviceName='exit')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome_2" ---
    # create an object to store info about Routine welcome_2
    welcome_2 = data.Routine(
        name='welcome_2',
        components=[welcome_text, welcome_end],
    )
    welcome_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for welcome_end
    welcome_end.keys = []
    welcome_end.rt = []
    _welcome_end_allKeys = []
    # store start times for welcome_2
    welcome_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome_2.tStart = globalClock.getTime(format='float')
    welcome_2.status = STARTED
    thisExp.addData('welcome_2.started', welcome_2.tStart)
    welcome_2.maxDuration = None
    # keep track of which components have finished
    welcome_2Components = welcome_2.components
    for thisComponent in welcome_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome_2" ---
    welcome_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *welcome_end* updates
        waitOnFlip = False
        
        # if welcome_end is starting this frame...
        if welcome_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_end.frameNStart = frameN  # exact frame index
            welcome_end.tStart = t  # local t and not account for scr refresh
            welcome_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_end.started')
            # update status
            welcome_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(welcome_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(welcome_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if welcome_end.status == STARTED and not waitOnFlip:
            theseKeys = welcome_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _welcome_end_allKeys.extend(theseKeys)
            if len(_welcome_end_allKeys):
                welcome_end.keys = _welcome_end_allKeys[-1].name  # just the last key pressed
                welcome_end.rt = _welcome_end_allKeys[-1].rt
                welcome_end.duration = _welcome_end_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome_2" ---
    for thisComponent in welcome_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome_2
    welcome_2.tStop = globalClock.getTime(format='float')
    welcome_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome_2.stopped', welcome_2.tStop)
    # check responses
    if welcome_end.keys in ['', [], None]:  # No response was made
        welcome_end.keys = None
    thisExp.addData('welcome_end.keys',welcome_end.keys)
    if welcome_end.keys != None:  # we had a response
        thisExp.addData('welcome_end.rt', welcome_end.rt)
        thisExp.addData('welcome_end.duration', welcome_end.duration)
    thisExp.nextEntry()
    # the Routine "welcome_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruc" ---
    # create an object to store info about Routine instruc
    instruc = data.Routine(
        name='instruc',
        components=[explain_participant, end_instructions],
    )
    instruc.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_instructions
    end_instructions.keys = []
    end_instructions.rt = []
    _end_instructions_allKeys = []
    # store start times for instruc
    instruc.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruc.tStart = globalClock.getTime(format='float')
    instruc.status = STARTED
    thisExp.addData('instruc.started', instruc.tStart)
    instruc.maxDuration = None
    # keep track of which components have finished
    instrucComponents = instruc.components
    for thisComponent in instruc.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruc" ---
    instruc.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *explain_participant* updates
        
        # if explain_participant is starting this frame...
        if explain_participant.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            explain_participant.frameNStart = frameN  # exact frame index
            explain_participant.tStart = t  # local t and not account for scr refresh
            explain_participant.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(explain_participant, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'explain_participant.started')
            # update status
            explain_participant.status = STARTED
            explain_participant.setAutoDraw(True)
        
        # if explain_participant is active this frame...
        if explain_participant.status == STARTED:
            # update params
            pass
        
        # *end_instructions* updates
        waitOnFlip = False
        
        # if end_instructions is starting this frame...
        if end_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_instructions.frameNStart = frameN  # exact frame index
            end_instructions.tStart = t  # local t and not account for scr refresh
            end_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_instructions.started')
            # update status
            end_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_instructions.status == STARTED and not waitOnFlip:
            theseKeys = end_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_instructions_allKeys.extend(theseKeys)
            if len(_end_instructions_allKeys):
                end_instructions.keys = _end_instructions_allKeys[-1].name  # just the last key pressed
                end_instructions.rt = _end_instructions_allKeys[-1].rt
                end_instructions.duration = _end_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instruc.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruc.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruc" ---
    for thisComponent in instruc.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruc
    instruc.tStop = globalClock.getTime(format='float')
    instruc.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruc.stopped', instruc.tStop)
    # check responses
    if end_instructions.keys in ['', [], None]:  # No response was made
        end_instructions.keys = None
    thisExp.addData('end_instructions.keys',end_instructions.keys)
    if end_instructions.keys != None:  # we had a response
        thisExp.addData('end_instructions.rt', end_instructions.rt)
        thisExp.addData('end_instructions.duration', end_instructions.duration)
    thisExp.nextEntry()
    # the Routine "instruc" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    series = data.TrialHandler2(
        name='series',
        nReps=N_RAMPS, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(series)  # add the loop to the experiment
    thisSerie = series.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSerie.rgb)
    if thisSerie != None:
        for paramName in thisSerie:
            globals()[paramName] = thisSerie[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSerie in series:
        currentLoop = series
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSerie.rgb)
        if thisSerie != None:
            for paramName in thisSerie:
                globals()[paramName] = thisSerie[paramName]
        
        # --- Prepare to start Routine "initiation_2" ---
        # create an object to store info about Routine initiation_2
        initiation_2 = data.Routine(
            name='initiation_2',
            components=[verification, end_verification],
        )
        initiation_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for end_verification
        end_verification.keys = []
        end_verification.rt = []
        _end_verification_allKeys = []
        # Run 'Begin Routine' code from reset_ma
        curr_item = -1 
        curr_item_loc = -1
        # store start times for initiation_2
        initiation_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        initiation_2.tStart = globalClock.getTime(format='float')
        initiation_2.status = STARTED
        thisExp.addData('initiation_2.started', initiation_2.tStart)
        initiation_2.maxDuration = None
        # keep track of which components have finished
        initiation_2Components = initiation_2.components
        for thisComponent in initiation_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "initiation_2" ---
        # if trial has changed, end Routine now
        if isinstance(series, data.TrialHandler2) and thisSerie.thisN != series.thisTrial.thisN:
            continueRoutine = False
        initiation_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *verification* updates
            
            # if verification is starting this frame...
            if verification.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                verification.frameNStart = frameN  # exact frame index
                verification.tStart = t  # local t and not account for scr refresh
                verification.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(verification, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'verification.started')
                # update status
                verification.status = STARTED
                verification.setAutoDraw(True)
            
            # if verification is active this frame...
            if verification.status == STARTED:
                # update params
                pass
            
            # *end_verification* updates
            waitOnFlip = False
            
            # if end_verification is starting this frame...
            if end_verification.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_verification.frameNStart = frameN  # exact frame index
                end_verification.tStart = t  # local t and not account for scr refresh
                end_verification.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_verification, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_verification.started')
                # update status
                end_verification.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_verification.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_verification.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if end_verification.status == STARTED and not waitOnFlip:
                theseKeys = end_verification.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_verification_allKeys.extend(theseKeys)
                if len(_end_verification_allKeys):
                    end_verification.keys = _end_verification_allKeys[-1].name  # just the last key pressed
                    end_verification.rt = _end_verification_allKeys[-1].rt
                    end_verification.duration = _end_verification_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                initiation_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in initiation_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "initiation_2" ---
        for thisComponent in initiation_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for initiation_2
        initiation_2.tStop = globalClock.getTime(format='float')
        initiation_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('initiation_2.stopped', initiation_2.tStop)
        # check responses
        if end_verification.keys in ['', [], None]:  # No response was made
            end_verification.keys = None
        series.addData('end_verification.keys',end_verification.keys)
        if end_verification.keys != None:  # we had a response
            series.addData('end_verification.rt', end_verification.rt)
            series.addData('end_verification.duration', end_verification.duration)
        # the Routine "initiation_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        ramp = data.TrialHandler2(
            name='ramp',
            nReps=20.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=[None], 
            seed=None, 
        )
        thisExp.addLoop(ramp)  # add the loop to the experiment
        thisRamp = ramp.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisRamp.rgb)
        if thisRamp != None:
            for paramName in thisRamp:
                globals()[paramName] = thisRamp[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisRamp in ramp:
            currentLoop = ramp
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisRamp.rgb)
            if thisRamp != None:
                for paramName in thisRamp:
                    globals()[paramName] = thisRamp[paramName]
            
            # --- Prepare to start Routine "stimulation_start" ---
            # create an object to store info about Routine stimulation_start
            stimulation_start = data.Routine(
                name='stimulation_start',
                components=[next_stim, confirm_continue],
            )
            stimulation_start.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from stim_start
            # Update counter
            curr_item += 1
            
            # Get current electrical intensity in mA
            intensity = float(electrical_intensity[curr_item])
            curr_ma = intensity
            
            # Nothing is pre-loaded on the DS5. The waveform is generated and fired later.
            fired = 0
            
            # Jitter waiting
            wait1jitter = np.random.choice([2000, 2100, 2200, 2300, 2400, 2500])
            wait1jitter = wait1jitter / 1000
            
            # Get phase
            ramp_phase = 1
            
            # Optional label replacing thermode localisation
            curr_item_loc += 1
            ds5_localisation = "DS5"
            # create starting attributes for confirm_continue
            confirm_continue.keys = []
            confirm_continue.rt = []
            _confirm_continue_allKeys = []
            # store start times for stimulation_start
            stimulation_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            stimulation_start.tStart = globalClock.getTime(format='float')
            stimulation_start.status = STARTED
            thisExp.addData('stimulation_start.started', stimulation_start.tStart)
            stimulation_start.maxDuration = None
            # keep track of which components have finished
            stimulation_startComponents = stimulation_start.components
            for thisComponent in stimulation_start.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "stimulation_start" ---
            # if trial has changed, end Routine now
            if isinstance(ramp, data.TrialHandler2) and thisRamp.thisN != ramp.thisTrial.thisN:
                continueRoutine = False
            stimulation_start.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *next_stim* updates
                
                # if next_stim is starting this frame...
                if next_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    next_stim.frameNStart = frameN  # exact frame index
                    next_stim.tStart = t  # local t and not account for scr refresh
                    next_stim.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(next_stim, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'next_stim.started')
                    # update status
                    next_stim.status = STARTED
                    next_stim.setAutoDraw(True)
                
                # if next_stim is active this frame...
                if next_stim.status == STARTED:
                    # update params
                    pass
                
                # *confirm_continue* updates
                waitOnFlip = False
                
                # if confirm_continue is starting this frame...
                if confirm_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confirm_continue.frameNStart = frameN  # exact frame index
                    confirm_continue.tStart = t  # local t and not account for scr refresh
                    confirm_continue.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confirm_continue, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'confirm_continue.started')
                    # update status
                    confirm_continue.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(confirm_continue.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(confirm_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if confirm_continue.status == STARTED and not waitOnFlip:
                    theseKeys = confirm_continue.getKeys(keyList=['p','n'], ignoreKeys=["escape"], waitRelease=False)
                    _confirm_continue_allKeys.extend(theseKeys)
                    if len(_confirm_continue_allKeys):
                        confirm_continue.keys = _confirm_continue_allKeys[-1].name  # just the last key pressed
                        confirm_continue.rt = _confirm_continue_allKeys[-1].rt
                        confirm_continue.duration = _confirm_continue_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    stimulation_start.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in stimulation_start.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "stimulation_start" ---
            for thisComponent in stimulation_start.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for stimulation_start
            stimulation_start.tStop = globalClock.getTime(format='float')
            stimulation_start.tStopRefresh = tThisFlipGlobal
            thisExp.addData('stimulation_start.stopped', stimulation_start.tStop)
            # Run 'End Routine' code from stim_start
            ramp.addData("intensity_ma", intensity)
            ramp.addData("noise_condition", "none")
            ramp.addData("wait1_dur", wait1jitter)
            
            if "q" in confirm_continue.keys:
                ramp.finished = 1
            # check responses
            if confirm_continue.keys in ['', [], None]:  # No response was made
                confirm_continue.keys = None
            ramp.addData('confirm_continue.keys',confirm_continue.keys)
            if confirm_continue.keys != None:  # we had a response
                ramp.addData('confirm_continue.rt', confirm_continue.rt)
                ramp.addData('confirm_continue.duration', confirm_continue.duration)
            # the Routine "stimulation_start" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "jitter_cross" ---
            # create an object to store info about Routine jitter_cross
            jitter_cross = data.Routine(
                name='jitter_cross',
                components=[jitter_cross_component],
            )
            jitter_cross.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for jitter_cross
            jitter_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            jitter_cross.tStart = globalClock.getTime(format='float')
            jitter_cross.status = STARTED
            thisExp.addData('jitter_cross.started', jitter_cross.tStart)
            jitter_cross.maxDuration = None
            # keep track of which components have finished
            jitter_crossComponents = jitter_cross.components
            for thisComponent in jitter_cross.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "jitter_cross" ---
            # if trial has changed, end Routine now
            if isinstance(ramp, data.TrialHandler2) and thisRamp.thisN != ramp.thisTrial.thisN:
                continueRoutine = False
            jitter_cross.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *jitter_cross_component* updates
                
                # if jitter_cross_component is starting this frame...
                if jitter_cross_component.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    jitter_cross_component.frameNStart = frameN  # exact frame index
                    jitter_cross_component.tStart = t  # local t and not account for scr refresh
                    jitter_cross_component.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(jitter_cross_component, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'jitter_cross_component.started')
                    # update status
                    jitter_cross_component.status = STARTED
                    jitter_cross_component.setAutoDraw(True)
                
                # if jitter_cross_component is active this frame...
                if jitter_cross_component.status == STARTED:
                    # update params
                    pass
                
                # if jitter_cross_component is stopping this frame...
                if jitter_cross_component.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > jitter_cross_component.tStartRefresh + wait1jitter-frameTolerance:
                        # keep track of stop time/frame for later
                        jitter_cross_component.tStop = t  # not accounting for scr refresh
                        jitter_cross_component.tStopRefresh = tThisFlipGlobal  # on global time
                        jitter_cross_component.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'jitter_cross_component.stopped')
                        # update status
                        jitter_cross_component.status = FINISHED
                        jitter_cross_component.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    jitter_cross.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in jitter_cross.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "jitter_cross" ---
            for thisComponent in jitter_cross.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for jitter_cross
            jitter_cross.tStop = globalClock.getTime(format='float')
            jitter_cross.tStopRefresh = tThisFlipGlobal
            thisExp.addData('jitter_cross.stopped', jitter_cross.tStop)
            # the Routine "jitter_cross" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "fire_electrical" ---
            # create an object to store info about Routine fire_electrical
            fire_electrical = data.Routine(
                name='fire_electrical',
                components=[cross_fire],
            )
            fire_electrical.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            
            
            # store start times for fire_electrical
            fire_electrical.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fire_electrical.tStart = globalClock.getTime(format='float')
            fire_electrical.status = STARTED
            thisExp.addData('fire_electrical.started', fire_electrical.tStart)
            fire_electrical.maxDuration = None
            # keep track of which components have finished
            fire_electricalComponents = fire_electrical.components
            for thisComponent in fire_electrical.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fire_electrical" ---
            # if trial has changed, end Routine now
            if isinstance(ramp, data.TrialHandler2) and thisRamp.thisN != ramp.thisTrial.thisN:
                continueRoutine = False
            fire_electrical.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from code
                if fired == 0 and frameN >= 1:
                    fired = 1
                    trial_label = (
                        f"{expInfo['participant']}_ds5_calib_ramp_"
                        f"{series.thisN:02d}_trial_{ramp.thisN:02d}_{intensity:.3f}mA"
                    )
                    ds5_metadata = fire_none_uncertainty_pulse(
                        intensity_ma=intensity,
                        config=ds5_config,
                        output_dir=waveform_dir,
                        trial_label=trial_label,
                    )
                
                # *cross_fire* updates
                
                # if cross_fire is starting this frame...
                if cross_fire.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cross_fire.frameNStart = frameN  # exact frame index
                    cross_fire.tStart = t  # local t and not account for scr refresh
                    cross_fire.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cross_fire, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_fire.started')
                    # update status
                    cross_fire.status = STARTED
                    cross_fire.setAutoDraw(True)
                
                # if cross_fire is active this frame...
                if cross_fire.status == STARTED:
                    # update params
                    pass
                
                # if cross_fire is stopping this frame...
                if cross_fire.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cross_fire.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        cross_fire.tStop = t  # not accounting for scr refresh
                        cross_fire.tStopRefresh = tThisFlipGlobal  # on global time
                        cross_fire.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cross_fire.stopped')
                        # update status
                        cross_fire.status = FINISHED
                        cross_fire.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    fire_electrical.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fire_electrical.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fire_electrical" ---
            for thisComponent in fire_electrical.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fire_electrical
            fire_electrical.tStop = globalClock.getTime(format='float')
            fire_electrical.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fire_electrical.stopped', fire_electrical.tStop)
            # Run 'End Routine' code from code
            thisExp.addData("ds5_fired", fired)
            thisExp.addData("noise_condition", "none")
            if fired:
                for key, value in ds5_metadata.items():
                    thisExp.addData(key, value)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if fire_electrical.maxDurationReached:
                routineTimer.addTime(-fire_electrical.maxDuration)
            elif fire_electrical.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "rating_scale" ---
            # create an object to store info about Routine rating_scale
            rating_scale = data.Routine(
                name='rating_scale',
                components=[slider_pain_2, slider_pain_resp_2, main_prompt_2, space_prompt_2],
            )
            rating_scale.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from slider_pain_code_2
            
            
            slider_pain_2.marker = visual.Rect(win, width=0.01, height=0.1, lineColor='red', fillColor='red', units=slider_pain_2.units)
            
            for i, line in enumerate(slider_pain_2.tickLines.sizes):
                slider_pain_2.tickLines.sizes[i][1] = 0
            slider_pain_2.tickLines._needVertexUpdate = True
            pos = np.random.uniform(0, 100)
            slider_pain_2.reset()
            slider_pain_2.startValue = pos
            
            # Start a timer for max duration
            vas_timer = core.Clock()
            vas_timer.reset()
            slider_pain_2.reset()
            # create starting attributes for slider_pain_resp_2
            slider_pain_resp_2.keys = []
            slider_pain_resp_2.rt = []
            _slider_pain_resp_2_allKeys = []
            # store start times for rating_scale
            rating_scale.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            rating_scale.tStart = globalClock.getTime(format='float')
            rating_scale.status = STARTED
            thisExp.addData('rating_scale.started', rating_scale.tStart)
            rating_scale.maxDuration = None
            # keep track of which components have finished
            rating_scaleComponents = rating_scale.components
            for thisComponent in rating_scale.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "rating_scale" ---
            # if trial has changed, end Routine now
            if isinstance(ramp, data.TrialHandler2) and thisRamp.thisN != ramp.thisTrial.thisN:
                continueRoutine = False
            rating_scale.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from slider_pain_code_2
                
                inc = 0.75
                wasmoved = 0
                
                while True:
                    keys = slider_pain_resp_2.getKeys(['n', 'm', 'space', 'escape'], waitRelease=False, clear=False)
                    if keys and not keys[-1].duration:
                        key = keys[-1].name
                        if 'n' == key:
                            slider_pain_2.markerPos -= inc
                            wasmoved = 1
                        if 'm' == key:
                            slider_pain_2.markerPos  += inc
                            wasmoved = 1
                
                        # check for quit (typically the Esc key)
                        if "escape" ==  key:
                            core.quit()
                            
                        if 'space' == key:
                            rating = slider_pain_2.markerPos
                            ratings_list.append(rating)
                            core.wait(0.1)
                            vaspressed = 1
                            thisExp.addData('VAS_rating', rating)
                            continueRoutine=False
                            break
                        time_elapsed = vas_timer.getTime()  # seconds since reset
                        if time_elapsed >= 10.0:  # after 10s, end routine we can change this to anything we want
                            rating = slider_pain_2.markerPos
                            ratings_list.append(rating)
                            core.wait(0.1)
                            vaspressed = 0
                            continueRoutine = False
                            break
                
                
                    #print(kb.state)
                    slider_pain_2.draw()
                    main_prompt_2.draw()
                    space_prompt_2.draw()
                    win.mouseVisible = False
                    win.flip()
                
                # *slider_pain_2* updates
                
                # if slider_pain_2 is starting this frame...
                if slider_pain_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    slider_pain_2.frameNStart = frameN  # exact frame index
                    slider_pain_2.tStart = t  # local t and not account for scr refresh
                    slider_pain_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_pain_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_pain_2.started')
                    # update status
                    slider_pain_2.status = STARTED
                    slider_pain_2.setAutoDraw(True)
                
                # if slider_pain_2 is active this frame...
                if slider_pain_2.status == STARTED:
                    # update params
                    pass
                
                # Check slider_pain_2 for response to end Routine
                if slider_pain_2.getRating() is not None and slider_pain_2.status == STARTED:
                    continueRoutine = False
                
                # *slider_pain_resp_2* updates
                waitOnFlip = False
                
                # if slider_pain_resp_2 is starting this frame...
                if slider_pain_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    slider_pain_resp_2.frameNStart = frameN  # exact frame index
                    slider_pain_resp_2.tStart = t  # local t and not account for scr refresh
                    slider_pain_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(slider_pain_resp_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'slider_pain_resp_2.started')
                    # update status
                    slider_pain_resp_2.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(slider_pain_resp_2.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(slider_pain_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if slider_pain_resp_2.status == STARTED and not waitOnFlip:
                    theseKeys = slider_pain_resp_2.getKeys(keyList=['n','m','enter', 'space'], ignoreKeys=["escape"], waitRelease=False)
                    _slider_pain_resp_2_allKeys.extend(theseKeys)
                    if len(_slider_pain_resp_2_allKeys):
                        slider_pain_resp_2.keys = _slider_pain_resp_2_allKeys[-1].name  # just the last key pressed
                        slider_pain_resp_2.rt = _slider_pain_resp_2_allKeys[-1].rt
                        slider_pain_resp_2.duration = _slider_pain_resp_2_allKeys[-1].duration
                
                # *main_prompt_2* updates
                
                # if main_prompt_2 is starting this frame...
                if main_prompt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    main_prompt_2.frameNStart = frameN  # exact frame index
                    main_prompt_2.tStart = t  # local t and not account for scr refresh
                    main_prompt_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(main_prompt_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'main_prompt_2.started')
                    # update status
                    main_prompt_2.status = STARTED
                    main_prompt_2.setAutoDraw(True)
                
                # if main_prompt_2 is active this frame...
                if main_prompt_2.status == STARTED:
                    # update params
                    pass
                
                # *space_prompt_2* updates
                
                # if space_prompt_2 is starting this frame...
                if space_prompt_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    space_prompt_2.frameNStart = frameN  # exact frame index
                    space_prompt_2.tStart = t  # local t and not account for scr refresh
                    space_prompt_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(space_prompt_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'space_prompt_2.started')
                    # update status
                    space_prompt_2.status = STARTED
                    space_prompt_2.setAutoDraw(True)
                
                # if space_prompt_2 is active this frame...
                if space_prompt_2.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    rating_scale.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in rating_scale.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "rating_scale" ---
            for thisComponent in rating_scale.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for rating_scale
            rating_scale.tStop = globalClock.getTime(format='float')
            rating_scale.tStopRefresh = tThisFlipGlobal
            thisExp.addData('rating_scale.stopped', rating_scale.tStop)
            ramp.addData('slider_pain_2.response', slider_pain_2.getRating())
            ramp.addData('slider_pain_2.rt', slider_pain_2.getRT())
            # the Routine "rating_scale" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "confirm_continue_2" ---
            # create an object to store info about Routine confirm_continue_2
            confirm_continue_2 = data.Routine(
                name='confirm_continue_2',
                components=[confirm_text, confirm_resp],
            )
            confirm_continue_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for confirm_resp
            confirm_resp.keys = []
            confirm_resp.rt = []
            _confirm_resp_allKeys = []
            # store start times for confirm_continue_2
            confirm_continue_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            confirm_continue_2.tStart = globalClock.getTime(format='float')
            confirm_continue_2.status = STARTED
            thisExp.addData('confirm_continue_2.started', confirm_continue_2.tStart)
            confirm_continue_2.maxDuration = None
            # keep track of which components have finished
            confirm_continue_2Components = confirm_continue_2.components
            for thisComponent in confirm_continue_2.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "confirm_continue_2" ---
            # if trial has changed, end Routine now
            if isinstance(ramp, data.TrialHandler2) and thisRamp.thisN != ramp.thisTrial.thisN:
                continueRoutine = False
            confirm_continue_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *confirm_text* updates
                
                # if confirm_text is starting this frame...
                if confirm_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confirm_text.frameNStart = frameN  # exact frame index
                    confirm_text.tStart = t  # local t and not account for scr refresh
                    confirm_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confirm_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'confirm_text.started')
                    # update status
                    confirm_text.status = STARTED
                    confirm_text.setAutoDraw(True)
                
                # if confirm_text is active this frame...
                if confirm_text.status == STARTED:
                    # update params
                    pass
                
                # *confirm_resp* updates
                waitOnFlip = False
                
                # if confirm_resp is starting this frame...
                if confirm_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    confirm_resp.frameNStart = frameN  # exact frame index
                    confirm_resp.tStart = t  # local t and not account for scr refresh
                    confirm_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(confirm_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'confirm_resp.started')
                    # update status
                    confirm_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(confirm_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(confirm_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if confirm_resp.status == STARTED and not waitOnFlip:
                    theseKeys = confirm_resp.getKeys(keyList=['y','n'], ignoreKeys=["escape"], waitRelease=False)
                    _confirm_resp_allKeys.extend(theseKeys)
                    if len(_confirm_resp_allKeys):
                        confirm_resp.keys = _confirm_resp_allKeys[-1].name  # just the last key pressed
                        confirm_resp.rt = _confirm_resp_allKeys[-1].rt
                        confirm_resp.duration = _confirm_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    confirm_continue_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in confirm_continue_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "confirm_continue_2" ---
            for thisComponent in confirm_continue_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for confirm_continue_2
            confirm_continue_2.tStop = globalClock.getTime(format='float')
            confirm_continue_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('confirm_continue_2.stopped', confirm_continue_2.tStop)
            # Run 'End Routine' code from confirm_code
            series_trial.append(series.thisN)
            ramp_trial.append(ramp.thisN)
            intensities_list.append(float(intensity))
            fired_list.append(int(fired))
            response = confirm_resp.keys
            if isinstance(response, list):
                response = response[-1] if response else None
            
            # Save only the last intensity they felt under the last ramp.
            if response == "y" and series.thisN == N_RAMPS - 1 and fired == 1:
                last_ramp_tolerated_ma = float(intensity)
            
            if response == "n":
                ramp.finished = 1
            
            if float(intensity) >= RAMP_MAX_MA:
                Ramp.finished = 1
            
            if curr_item_loc >= 3:
                curr_item_loc = -1
            # check responses
            if confirm_resp.keys in ['', [], None]:  # No response was made
                confirm_resp.keys = None
            ramp.addData('confirm_resp.keys',confirm_resp.keys)
            if confirm_resp.keys != None:  # we had a response
                ramp.addData('confirm_resp.rt', confirm_resp.rt)
                ramp.addData('confirm_resp.duration', confirm_resp.duration)
            # the Routine "confirm_continue_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 20.0 repeats of 'ramp'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed N_RAMPS repeats of 'series'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "compile_ramps" ---
    # create an object to store info about Routine compile_ramps
    compile_ramps = data.Routine(
        name='compile_ramps',
        components=[first_over, continue_first_over],
    )
    compile_ramps.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from first_over_compilation
    ramp_data = {
        "series": series_trial,
        "trial": ramp_trial,
        "intensity_ma": intensities_list,
        "rating": ratings_list,
        "fired": fired_list,
    }
    
    print("ramp_data is:", ramp_data)
    ramp_data = pd.DataFrame(data=ramp_data)
    
    calib_dir.mkdir(parents=True, exist_ok=True)
    filenameramp = calib_dir / f"{expInfo['participant']}_{expName}_{expInfo['date']}_ramp_ratings.csv"
    ramp_data.to_csv(filenameramp, index=False)
    
    # Use only last ramp.
    ramp_data_last = ramp_data[ramp_data.series == N_RAMPS - 1]
    ramp_data_last = ramp_data_last[ramp_data_last["fired"] == 1]
    
    # Tolerance/high value: last intensity they accepted in the final ramp only.
    if last_ramp_tolerated_ma is None:
        tolerance = float(np.max(ramp_data_last["intensity_ma"]))
    else:
        tolerance = float(last_ramp_tolerated_ma)
    
    # Threshold: first final-ramp intensity with rating > 0.
    ramp_data_above = ramp_data_last[ramp_data_last["rating"] > 0]
    if len(ramp_data_above) == 0:
        threshold = float(RAMP_START_MA)
    else:
        threshold = float(np.min(ramp_data_above["intensity_ma"]))
    
    ramp_data["tolerance_ma"] = tolerance
    ramp_data["threshold_ma"] = threshold
    ramp_data["last_ramp_tolerated_ma"] = last_ramp_tolerated_ma
    ramp_data.to_csv(filenameramp, index=False)
    
    # Generate fixed second-phase electrical intensities from threshold to tolerance.
    rand_intensities = np.around(np.linspace(threshold, tolerance, 14), 3)
    np.random.shuffle(rand_intensities)
    # create starting attributes for continue_first_over
    continue_first_over.keys = []
    continue_first_over.rt = []
    _continue_first_over_allKeys = []
    # store start times for compile_ramps
    compile_ramps.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    compile_ramps.tStart = globalClock.getTime(format='float')
    compile_ramps.status = STARTED
    thisExp.addData('compile_ramps.started', compile_ramps.tStart)
    compile_ramps.maxDuration = None
    # keep track of which components have finished
    compile_rampsComponents = compile_ramps.components
    for thisComponent in compile_ramps.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "compile_ramps" ---
    compile_ramps.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *first_over* updates
        
        # if first_over is starting this frame...
        if first_over.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            first_over.frameNStart = frameN  # exact frame index
            first_over.tStart = t  # local t and not account for scr refresh
            first_over.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(first_over, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'first_over.started')
            # update status
            first_over.status = STARTED
            first_over.setAutoDraw(True)
        
        # if first_over is active this frame...
        if first_over.status == STARTED:
            # update params
            pass
        
        # *continue_first_over* updates
        waitOnFlip = False
        
        # if continue_first_over is starting this frame...
        if continue_first_over.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            continue_first_over.frameNStart = frameN  # exact frame index
            continue_first_over.tStart = t  # local t and not account for scr refresh
            continue_first_over.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(continue_first_over, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'continue_first_over.started')
            # update status
            continue_first_over.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(continue_first_over.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(continue_first_over.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if continue_first_over.status == STARTED and not waitOnFlip:
            theseKeys = continue_first_over.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _continue_first_over_allKeys.extend(theseKeys)
            if len(_continue_first_over_allKeys):
                continue_first_over.keys = _continue_first_over_allKeys[-1].name  # just the last key pressed
                continue_first_over.rt = _continue_first_over_allKeys[-1].rt
                continue_first_over.duration = _continue_first_over_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            compile_ramps.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in compile_ramps.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "compile_ramps" ---
    for thisComponent in compile_ramps.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for compile_ramps
    compile_ramps.tStop = globalClock.getTime(format='float')
    compile_ramps.tStopRefresh = tThisFlipGlobal
    thisExp.addData('compile_ramps.stopped', compile_ramps.tStop)
    # check responses
    if continue_first_over.keys in ['', [], None]:  # No response was made
        continue_first_over.keys = None
    thisExp.addData('continue_first_over.keys',continue_first_over.keys)
    if continue_first_over.keys != None:  # we had a response
        thisExp.addData('continue_first_over.rt', continue_first_over.rt)
        thisExp.addData('continue_first_over.duration', continue_first_over.duration)
    thisExp.nextEntry()
    # the Routine "compile_ramps" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "second_phase_start" ---
    # create an object to store info about Routine second_phase_start
    second_phase_start = data.Routine(
        name='second_phase_start',
        components=[second_instructions, start_second_p],
    )
    second_phase_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_second
    # Get intensities
    intensity_list = rand_intensities
    lenght = len(intensity_list)
    # create starting attributes for start_second_p
    start_second_p.keys = []
    start_second_p.rt = []
    _start_second_p_allKeys = []
    # store start times for second_phase_start
    second_phase_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    second_phase_start.tStart = globalClock.getTime(format='float')
    second_phase_start.status = STARTED
    thisExp.addData('second_phase_start.started', second_phase_start.tStart)
    second_phase_start.maxDuration = None
    # keep track of which components have finished
    second_phase_startComponents = second_phase_start.components
    for thisComponent in second_phase_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "second_phase_start" ---
    second_phase_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *second_instructions* updates
        
        # if second_instructions is starting this frame...
        if second_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            second_instructions.frameNStart = frameN  # exact frame index
            second_instructions.tStart = t  # local t and not account for scr refresh
            second_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(second_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'second_instructions.started')
            # update status
            second_instructions.status = STARTED
            second_instructions.setAutoDraw(True)
        
        # if second_instructions is active this frame...
        if second_instructions.status == STARTED:
            # update params
            pass
        
        # *start_second_p* updates
        waitOnFlip = False
        
        # if start_second_p is starting this frame...
        if start_second_p.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_second_p.frameNStart = frameN  # exact frame index
            start_second_p.tStart = t  # local t and not account for scr refresh
            start_second_p.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_second_p, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_second_p.started')
            # update status
            start_second_p.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_second_p.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_second_p.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_second_p.status == STARTED and not waitOnFlip:
            theseKeys = start_second_p.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _start_second_p_allKeys.extend(theseKeys)
            if len(_start_second_p_allKeys):
                start_second_p.keys = _start_second_p_allKeys[-1].name  # just the last key pressed
                start_second_p.rt = _start_second_p_allKeys[-1].rt
                start_second_p.duration = _start_second_p_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            second_phase_start.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in second_phase_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "second_phase_start" ---
    for thisComponent in second_phase_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for second_phase_start
    second_phase_start.tStop = globalClock.getTime(format='float')
    second_phase_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('second_phase_start.stopped', second_phase_start.tStop)
    # check responses
    if start_second_p.keys in ['', [], None]:  # No response was made
        start_second_p.keys = None
    thisExp.addData('start_second_p.keys',start_second_p.keys)
    if start_second_p.keys != None:  # we had a response
        thisExp.addData('start_second_p.rt', start_second_p.rt)
        thisExp.addData('start_second_p.duration', start_second_p.duration)
    thisExp.nextEntry()
    # the Routine "second_phase_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=lenght, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "trials_part_2" ---
        # create an object to store info about Routine trials_part_2
        trials_part_2 = data.Routine(
            name='trials_part_2',
            components=[admnisiter_stim_2, admin_2],
        )
        trials_part_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from admin_2_start
               
        # Get the intensity
        curr_item_scd += 1
        valeur_temperature = intensity_list[curr_item_scd]
        
        fired = 0
        
        trials_phase = 0
        wait1jitter = np.random.choice([2000, 2100, 2200, 2300, 2400, 2500])
        wait1jitter = wait1jitter / 1000
        trials.addData("wait1_dur", wait1jitter)
        
        
        intensity = float(intensity_list[curr_item_scd])
        curr_ma = intensity
        fired = 0
        
        curr_item_loc += 1
        ds5_localisation = "DS5"
        
        
        # create starting attributes for admin_2
        admin_2.keys = []
        admin_2.rt = []
        _admin_2_allKeys = []
        # store start times for trials_part_2
        trials_part_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trials_part_2.tStart = globalClock.getTime(format='float')
        trials_part_2.status = STARTED
        thisExp.addData('trials_part_2.started', trials_part_2.tStart)
        trials_part_2.maxDuration = None
        # keep track of which components have finished
        trials_part_2Components = trials_part_2.components
        for thisComponent in trials_part_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials_part_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        trials_part_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *admnisiter_stim_2* updates
            
            # if admnisiter_stim_2 is starting this frame...
            if admnisiter_stim_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                admnisiter_stim_2.frameNStart = frameN  # exact frame index
                admnisiter_stim_2.tStart = t  # local t and not account for scr refresh
                admnisiter_stim_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(admnisiter_stim_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'admnisiter_stim_2.started')
                # update status
                admnisiter_stim_2.status = STARTED
                admnisiter_stim_2.setAutoDraw(True)
            
            # if admnisiter_stim_2 is active this frame...
            if admnisiter_stim_2.status == STARTED:
                # update params
                pass
            
            # *admin_2* updates
            waitOnFlip = False
            
            # if admin_2 is starting this frame...
            if admin_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                admin_2.frameNStart = frameN  # exact frame index
                admin_2.tStart = t  # local t and not account for scr refresh
                admin_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(admin_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'admin_2.started')
                # update status
                admin_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(admin_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(admin_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if admin_2.status == STARTED and not waitOnFlip:
                theseKeys = admin_2.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _admin_2_allKeys.extend(theseKeys)
                if len(_admin_2_allKeys):
                    admin_2.keys = _admin_2_allKeys[-1].name  # just the last key pressed
                    admin_2.rt = _admin_2_allKeys[-1].rt
                    admin_2.duration = _admin_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trials_part_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials_part_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials_part_2" ---
        for thisComponent in trials_part_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trials_part_2
        trials_part_2.tStop = globalClock.getTime(format='float')
        trials_part_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trials_part_2.stopped', trials_part_2.tStop)
        # Run 'End Routine' code from admin_2_start
        trials.addData("wait1_dur", wait1jitter)
        
        # check responses
        if admin_2.keys in ['', [], None]:  # No response was made
            admin_2.keys = None
        trials.addData('admin_2.keys',admin_2.keys)
        if admin_2.keys != None:  # we had a response
            trials.addData('admin_2.rt', admin_2.rt)
            trials.addData('admin_2.duration', admin_2.duration)
        # the Routine "trials_part_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "jitter_2" ---
        # create an object to store info about Routine jitter_2
        jitter_2 = data.Routine(
            name='jitter_2',
            components=[jitter_fix_text_2],
        )
        jitter_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for jitter_2
        jitter_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        jitter_2.tStart = globalClock.getTime(format='float')
        jitter_2.status = STARTED
        thisExp.addData('jitter_2.started', jitter_2.tStart)
        jitter_2.maxDuration = None
        # keep track of which components have finished
        jitter_2Components = jitter_2.components
        for thisComponent in jitter_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "jitter_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        jitter_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *jitter_fix_text_2* updates
            
            # if jitter_fix_text_2 is starting this frame...
            if jitter_fix_text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                jitter_fix_text_2.frameNStart = frameN  # exact frame index
                jitter_fix_text_2.tStart = t  # local t and not account for scr refresh
                jitter_fix_text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(jitter_fix_text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'jitter_fix_text_2.started')
                # update status
                jitter_fix_text_2.status = STARTED
                jitter_fix_text_2.setAutoDraw(True)
            
            # if jitter_fix_text_2 is active this frame...
            if jitter_fix_text_2.status == STARTED:
                # update params
                pass
            
            # if jitter_fix_text_2 is stopping this frame...
            if jitter_fix_text_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > jitter_fix_text_2.tStartRefresh + wait1jitter-frameTolerance:
                    # keep track of stop time/frame for later
                    jitter_fix_text_2.tStop = t  # not accounting for scr refresh
                    jitter_fix_text_2.tStopRefresh = tThisFlipGlobal  # on global time
                    jitter_fix_text_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'jitter_fix_text_2.stopped')
                    # update status
                    jitter_fix_text_2.status = FINISHED
                    jitter_fix_text_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                jitter_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in jitter_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "jitter_2" ---
        for thisComponent in jitter_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for jitter_2
        jitter_2.tStop = globalClock.getTime(format='float')
        jitter_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('jitter_2.stopped', jitter_2.tStop)
        # the Routine "jitter_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "send_electrical_2" ---
        # create an object to store info about Routine send_electrical_2
        send_electrical_2 = data.Routine(
            name='send_electrical_2',
            components=[fix_2],
        )
        send_electrical_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from electrical_2
        
        
        # store start times for send_electrical_2
        send_electrical_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        send_electrical_2.tStart = globalClock.getTime(format='float')
        send_electrical_2.status = STARTED
        thisExp.addData('send_electrical_2.started', send_electrical_2.tStart)
        send_electrical_2.maxDuration = None
        # keep track of which components have finished
        send_electrical_2Components = send_electrical_2.components
        for thisComponent in send_electrical_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "send_electrical_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        send_electrical_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from electrical_2
            if fired == 0 and frameN >= 1:
                fired = 1
                trial_label = (
                    f"{expInfo['participant']}_ds5_calib_random_"
                    f"{trials.thisN:02d}_{intensity:.3f}mA"
                )
                ds5_metadata = fire_none_uncertainty_pulse(
                    intensity_ma=intensity,
                    config=ds5_config,
                    output_dir=waveform_dir,
                    trial_label=trial_label,
                )
            
            # *fix_2* updates
            
            # if fix_2 is starting this frame...
            if fix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_2.frameNStart = frameN  # exact frame index
                fix_2.tStart = t  # local t and not account for scr refresh
                fix_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_2.started')
                # update status
                fix_2.status = STARTED
                fix_2.setAutoDraw(True)
            
            # if fix_2 is active this frame...
            if fix_2.status == STARTED:
                # update params
                pass
            
            # if fix_2 is stopping this frame...
            if fix_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_2.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_2.tStop = t  # not accounting for scr refresh
                    fix_2.tStopRefresh = tThisFlipGlobal  # on global time
                    fix_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_2.stopped')
                    # update status
                    fix_2.status = FINISHED
                    fix_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                send_electrical_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in send_electrical_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "send_electrical_2" ---
        for thisComponent in send_electrical_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for send_electrical_2
        send_electrical_2.tStop = globalClock.getTime(format='float')
        send_electrical_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('send_electrical_2.stopped', send_electrical_2.tStop)
        # Run 'End Routine' code from electrical_2
        thisExp.addData("ds5_fired", fired)
        thisExp.addData("noise_condition", "none")
        if fired:
            for key, value in ds5_metadata.items():
                thisExp.addData(key, value)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if send_electrical_2.maxDurationReached:
            routineTimer.addTime(-send_electrical_2.maxDuration)
        elif send_electrical_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "ratings_2" ---
        # create an object to store info about Routine ratings_2
        ratings_2 = data.Routine(
            name='ratings_2',
            components=[slider_pain_3, slider_pain_resp_3, main_prompt_3, space_prompt_3],
        )
        ratings_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from slider_pain_code_3
        
        
        slider_pain_3.marker = visual.Rect(win, width=0.01, height=0.1, lineColor='red', fillColor='red', units=slider_pain_3.units)
        
        for i, line in enumerate(slider_pain_3.tickLines.sizes):
            slider_pain_3.tickLines.sizes[i][1] = 0
        slider_pain_3.tickLines._needVertexUpdate = True
        pos = np.random.uniform(0, 100)
        slider_pain_3.reset()
        slider_pain_3.startValue = pos
        
        # Start a timer for max duration
        vas_timer = core.Clock()
        vas_timer.reset()
        slider_pain_3.reset()
        # create starting attributes for slider_pain_resp_3
        slider_pain_resp_3.keys = []
        slider_pain_resp_3.rt = []
        _slider_pain_resp_3_allKeys = []
        # store start times for ratings_2
        ratings_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ratings_2.tStart = globalClock.getTime(format='float')
        ratings_2.status = STARTED
        thisExp.addData('ratings_2.started', ratings_2.tStart)
        ratings_2.maxDuration = None
        # keep track of which components have finished
        ratings_2Components = ratings_2.components
        for thisComponent in ratings_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ratings_2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        ratings_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from slider_pain_code_3
            
            inc = 0.75
            wasmoved = 0
            
            while True:
                keys = slider_pain_resp_3.getKeys(['n', 'm', 'space', 'escape'], waitRelease=False, clear=False)
                if keys and not keys[-1].duration:
                    key = keys[-1].name
                    if 'n' == key:
                        slider_pain_3.markerPos -= inc
                        wasmoved = 1
                    if 'm' == key:
                        slider_pain_3.markerPos  += inc
                        wasmoved = 1
            
                    # check for quit (typically the Esc key)
                    if "escape" ==  key:
                        core.quit()
                        
                    if 'space' == key:
                        rating = slider_pain_3.markerPos
                        ratings_random.append(rating)
                        core.wait(0.1)
                        vaspressed = 1
                        thisExp.addData('VAS_rating', rating)
                        continueRoutine=False
                        break
                    time_elapsed = vas_timer.getTime()  # seconds since reset
                    if time_elapsed >= 10.0:  # after 10s, end routine we can change this to anything we want
                        rating = slider_pain_3.markerPos
                        ratings_random.append(rating)
                        core.wait(0.1)
                        vaspressed = 0
                        continueRoutine = False
                        break
            
            
                #print(kb.state)
                slider_pain_3.draw()
                main_prompt_3.draw()
                space_prompt_3.draw()
                win.mouseVisible = False
                win.flip()
            
            # *slider_pain_3* updates
            
            # if slider_pain_3 is starting this frame...
            if slider_pain_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_pain_3.frameNStart = frameN  # exact frame index
                slider_pain_3.tStart = t  # local t and not account for scr refresh
                slider_pain_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_pain_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_pain_3.started')
                # update status
                slider_pain_3.status = STARTED
                slider_pain_3.setAutoDraw(True)
            
            # if slider_pain_3 is active this frame...
            if slider_pain_3.status == STARTED:
                # update params
                pass
            
            # Check slider_pain_3 for response to end Routine
            if slider_pain_3.getRating() is not None and slider_pain_3.status == STARTED:
                continueRoutine = False
            
            # *slider_pain_resp_3* updates
            waitOnFlip = False
            
            # if slider_pain_resp_3 is starting this frame...
            if slider_pain_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_pain_resp_3.frameNStart = frameN  # exact frame index
                slider_pain_resp_3.tStart = t  # local t and not account for scr refresh
                slider_pain_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_pain_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_pain_resp_3.started')
                # update status
                slider_pain_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(slider_pain_resp_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(slider_pain_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if slider_pain_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = slider_pain_resp_3.getKeys(keyList=['n','m','enter', 'space'], ignoreKeys=["escape"], waitRelease=False)
                _slider_pain_resp_3_allKeys.extend(theseKeys)
                if len(_slider_pain_resp_3_allKeys):
                    slider_pain_resp_3.keys = _slider_pain_resp_3_allKeys[-1].name  # just the last key pressed
                    slider_pain_resp_3.rt = _slider_pain_resp_3_allKeys[-1].rt
                    slider_pain_resp_3.duration = _slider_pain_resp_3_allKeys[-1].duration
            
            # *main_prompt_3* updates
            
            # if main_prompt_3 is starting this frame...
            if main_prompt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                main_prompt_3.frameNStart = frameN  # exact frame index
                main_prompt_3.tStart = t  # local t and not account for scr refresh
                main_prompt_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(main_prompt_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'main_prompt_3.started')
                # update status
                main_prompt_3.status = STARTED
                main_prompt_3.setAutoDraw(True)
            
            # if main_prompt_3 is active this frame...
            if main_prompt_3.status == STARTED:
                # update params
                pass
            
            # *space_prompt_3* updates
            
            # if space_prompt_3 is starting this frame...
            if space_prompt_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                space_prompt_3.frameNStart = frameN  # exact frame index
                space_prompt_3.tStart = t  # local t and not account for scr refresh
                space_prompt_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(space_prompt_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'space_prompt_3.started')
                # update status
                space_prompt_3.status = STARTED
                space_prompt_3.setAutoDraw(True)
            
            # if space_prompt_3 is active this frame...
            if space_prompt_3.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                ratings_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ratings_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ratings_2" ---
        for thisComponent in ratings_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ratings_2
        ratings_2.tStop = globalClock.getTime(format='float')
        ratings_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('ratings_2.stopped', ratings_2.tStop)
        trials.addData('slider_pain_3.response', slider_pain_3.getRating())
        trials.addData('slider_pain_3.rt', slider_pain_3.getRT())
        # the Routine "ratings_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed lenght repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thank_you" ---
    # create an object to store info about Routine thank_you
    thank_you = data.Routine(
        name='thank_you',
        components=[thank_you_text, exit],
    )
    thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from end_code
    # Get data in a csv file
    ramp_data = {'intensity': intensity_list,
                  'rating': ratings_random}
    
    # Get tolerance level and offset a bit
    tolerance = np.max(intensity_list)
    
    # Création d'un DataFrame à partir du dictionnaire 'ramp_data'
    ramp_df = pd.DataFrame(ramp_data)
    print('ramp_df is : ', ramp_df)
    
    # Sélection des lignes avec un 'rating' positif
    ramp_data_above = ramp_df[ramp_df['rating'] > 0]
    
    # Obtenir le minimum d'intensité parmi les valeurs au-dessus du seuil
    threshold = np.min(ramp_data_above['intensity'])
    
    
    
    # Get min intensity above threshold
    #ramp_data_above = ramp_data[ramp_data['rating'] > 0]
    #threshold = np.min(ramp_data_above['intensity'])
    
    # Filtrer les données avec un rating positif
    #ramp_data_above = {
        #'intensity': [intensity for intensity, rating in zip(valeur_temperature, ratings_random) if rating > 0],
        #'rating': [rating for rating in ratings_random if rating > 0]}
    
    
    
    # create starting attributes for exit
    exit.keys = []
    exit.rt = []
    _exit_allKeys = []
    # store start times for thank_you
    thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thank_you.tStart = globalClock.getTime(format='float')
    thank_you.status = STARTED
    thisExp.addData('thank_you.started', thank_you.tStart)
    thank_you.maxDuration = None
    # keep track of which components have finished
    thank_youComponents = thank_you.components
    for thisComponent in thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thank_you_text* updates
        
        # if thank_you_text is starting this frame...
        if thank_you_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thank_you_text.frameNStart = frameN  # exact frame index
            thank_you_text.tStart = t  # local t and not account for scr refresh
            thank_you_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thank_you_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thank_you_text.started')
            # update status
            thank_you_text.status = STARTED
            thank_you_text.setAutoDraw(True)
        
        # if thank_you_text is active this frame...
        if thank_you_text.status == STARTED:
            # update params
            pass
        
        # if thank_you_text is stopping this frame...
        if thank_you_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thank_you_text.tStartRefresh + 15-frameTolerance:
                # keep track of stop time/frame for later
                thank_you_text.tStop = t  # not accounting for scr refresh
                thank_you_text.tStopRefresh = tThisFlipGlobal  # on global time
                thank_you_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thank_you_text.stopped')
                # update status
                thank_you_text.status = FINISHED
                thank_you_text.setAutoDraw(False)
        
        # *exit* updates
        waitOnFlip = False
        
        # if exit is starting this frame...
        if exit.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exit.frameNStart = frameN  # exact frame index
            exit.tStart = t  # local t and not account for scr refresh
            exit.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exit, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exit.started')
            # update status
            exit.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(exit.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(exit.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if exit.status == STARTED and not waitOnFlip:
            theseKeys = exit.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
            _exit_allKeys.extend(theseKeys)
            if len(_exit_allKeys):
                exit.keys = _exit_allKeys[-1].name  # just the last key pressed
                exit.rt = _exit_allKeys[-1].rt
                exit.duration = _exit_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thank_you.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thank_you
    thank_you.tStop = globalClock.getTime(format='float')
    thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thank_you.stopped', thank_you.tStop)
    # check responses
    if exit.keys in ['', [], None]:  # No response was made
        exit.keys = None
    thisExp.addData('exit.keys',exit.keys)
    if exit.keys != None:  # we had a response
        thisExp.addData('exit.rt', exit.rt)
        thisExp.addData('exit.duration', exit.duration)
    thisExp.nextEntry()
    # the Routine "thank_you" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # Run 'End Experiment' code from end_code
    xdata = [float(i) for i in intensity_list]
    ratings_random = [float(r) for r in ratings_random]
    
    target_rating = [0.3, 0.6]
    target_rating = [c * 100 for c in target_rating]
    
    if np.max(ratings_random) > 0:
        ratings_random = [(c / np.max(ratings_random)) * 100 for c in ratings_random]
    else:
        ratings_random = [0 for c in ratings_random]
    
    ydata = ratings_random
    intensities = np.arange(threshold, tolerance + 0.1, 0.1)
    rate_data = pd.DataFrame()
    rate_data['ydata'] = ydata
    rate_data['xdata'] = xdata
    
    filenameramp_png = _thisDir + os.sep + u'data/%s_%s/Calibration/%s_%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], 'fitted_values.png' )
    filenameramp_csv = _thisDir + os.sep + u'data/%s_%s/Calibration/%s_%s_%s_%s' % (expInfo['participant'], data.getDateStr(format="%Y-%m-%d"), expInfo['participant'], expName, expInfo['date'], 'fitted_values.csv' )
    
    
    def fit_model(func_type, params, levels):
        return {
            #Scale down by 0.1 to prevent overflow error
            "linear" : params[0] + params[1] * levels,
            "hyper"  : params[0] + params[1]/(params[2] + levels),
            "para"   : params[0] + params[1] * levels + params[2] * (levels**2),
            "expo"   : params[0] + params[1]*(levels**params[2])
            }[func_type] 
    
    def get_model_sse(params, rate_data, func):
    
        # mean center to remove intercept
        fitted = fit_model(func, params, rate_data['xdata'])
        sse = np.sum((rate_data['ydata'] - fitted)**2)
    
        return sse
    
    results = dict()
    models = ['linear', 'hyper', 'para', 'expo']
    for func in models:
        results[func] = dict()
        res = minimize(fun=get_model_sse, x0=[1, 1, 1],
                        args=(rate_data, func),
                        method='SLSQP',
                        options={'maxiter': 100})
    
        results[func]['fitted'] = fit_model(func, res['x'], intensities)
        results[func]['sse'] = res['fun']
        results[func]['params'] = res['x']
        results[func]['stim_val'] = []
        results[func]['fitted_val'] = []
    
        for t in target_rating:
            loc = np.argmin(np.abs(results[func]['fitted'] - t))
            results[func]['stim_val'].append(intensities[loc])
            results[func]['fitted_val'].append(results[func]['fitted'] [loc])
        # Save at each model in case crash
        pd.DataFrame(results).to_csv(filenameramp_csv)
        win_model = models[np.argmin([results[c]['sse'] for c in list(results.keys())])]
        results_temp = results.copy()
        results_temp['win_model'] = win_model
        results_temp['target_ratings'] = str(target_rating)
        results_temp['selected_intensities'] = str(results[win_model]['stim_val'])
        results_temp['extrapolated_ratings'] = str(results[win_model]['fitted_val'])
        pd.DataFrame(results_temp).to_csv(filenameramp_csv)
        # Plot
        calib_plot = plt.figure(figsize=(6, 5))
        plt.scatter(xdata, ydata, label='Actual ratings')
        plt.plot(intensities, results[win_model]['fitted'], label='Best fit', color='orange')
        plt.scatter(results[win_model]['stim_val'], results[win_model]['fitted_val'], label=win_model
                    + ' fitted values',
                    alpha=0.9, marker='^', s=60, color='orange')
        plt.plot(intensities, results['linear']['fitted'], label='Linear fit', color='green')
        plt.scatter(results['linear']['stim_val'], results['linear']['fitted_val'], label='Linearly spaced values',
                    alpha=0.5, color='green')
        plt.ylim(-1, 105)
        # plt.title(expInfo['participant'])
        plt.xlabel('Intensity (Celsius)')
        plt.ylabel('Rating (% max rating)')
        plt.legend()
        plt.savefig(filenameramp_png)
    
    win_model = models[np.argmin([results[c]['sse'] for c in list(results.keys())])]
    
    results['win_model'] = win_model
    results['target_ratings'] = str(target_rating)
    results['selected_intensities'] = str(results[win_model]['stim_val'])
    results['extrapolated_ratings'] = str(results[win_model]['fitted_val'])
    
    # Save final
    pd.DataFrame(results).to_csv(filenameramp_csv)
    # Plot
    calib_plot = plt.figure(figsize=(6, 5))
    plt.scatter(xdata, ydata, label='Actual ratings')
    plt.plot(intensities, results[win_model]['fitted'], label='Best fit', color='orange')
    plt.scatter(results[win_model]['stim_val'], results[win_model]['fitted_val'], label=win_model
                + ' fitted values',
                alpha=0.9, marker='^', s=60, color='orange')
    plt.plot(intensities, results['linear']['fitted'], label='Linear fit', color='green')
    plt.scatter(results['linear']['stim_val'], results['linear']['fitted_val'], label='Linearly spaced values',
                alpha=0.5, color='green')
    plt.ylim(-1, 105)
    # plt.title(expInfo['participant'])
    plt.xlabel('Intensity (Celsius)')
    plt.ylabel('Rating (% max rating)')
    plt.legend()
    plt.savefig(filenameramp_png)
    
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
