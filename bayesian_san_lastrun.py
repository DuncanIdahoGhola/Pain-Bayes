#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on March 03, 2026, at 13:36
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

# Run 'Before Experiment' code from boxes_creation
from psychopy import visual, core

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'bayesian_san'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': 'sub-000',
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
    filename = u'data/%s/bayes_pain/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\labmp\\Desktop\\Sanounou\\bayesian_san_lastrun.py',
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
    if deviceManager.getDevice('instr_leave') is None:
        # initialise instr_leave
        instr_leave = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_leave',
        )
    if deviceManager.getDevice('slider_pain_resp') is None:
        # initialise slider_pain_resp
        slider_pain_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='slider_pain_resp',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
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
    
    # --- Initialize components for Routine "set_up" ---
    # Run 'Begin Experiment' code from set_up_2
    participant_id = expInfo['participant']
    last_three_digits = participant_id[-3:]
    
    #we could use last_trhee_digits to counterbalance something if we wanted to here
    from pathlib import Path
    
    directory = Path(__file__).parent
    conditions_dir = directory / 'condition_files'
    
    cond_files = conditions_dir / f'{last_three_digits}_conditions.csv'
    cond_file = str(cond_files)
    
    
    # --- Initialize components for Routine "instructions" ---
    instr_text = visual.TextStim(win=win, name='instr_text',
        text='Add instructions here\n\npress spacebar to continue ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_leave = keyboard.Keyboard(deviceName='instr_leave')
    
    # --- Initialize components for Routine "fix" ---
    fixation_cross = visual.TextStim(win=win, name='fixation_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "vas_indicator" ---
    # Run 'Begin Experiment' code from pain_indicator_slider
    
    
    
    slider_indicator = visual.Slider(win=win, name='slider_indicator',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=['Aucune\ndouleur', 'Pire douleur\nimaginable'], ticks=[0, 100], granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.04,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    
    # --- Initialize components for Routine "pain_fix" ---
    fake_pain = visual.TextStim(win=win, name='fake_pain',
        text='pain stim goes here\nouch',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "vas_rating" ---
    slider_pain = visual.Slider(win=win, name='slider_pain',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=['Aucune\ndouleur', 'Pire douleur\nimaginable'], ticks=[0, 100], granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='White', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Arial', labelHeight=0.04,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    slider_pain_resp = keyboard.Keyboard(deviceName='slider_pain_resp')
    main_prompt = visual.TextStim(win=win, name='main_prompt',
        text="Veuillez évaluer l'intensité de la douleur ressentie.",
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    space_prompt = visual.TextStim(win=win, name='space_prompt',
        text="Appuyez sur la barre d'espacement pour valider.",
        font='Arial',
        pos=(0, -0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "end_exp" ---
    end_exp_text = visual.TextStim(win=win, name='end_exp_text',
        text='Thank you for participating in our experiment\nThe exp will close in 10 seconds, please wait \n\nChange this text for a custom thank you text',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
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
    
    # --- Prepare to start Routine "set_up" ---
    # create an object to store info about Routine set_up
    set_up = data.Routine(
        name='set_up',
        components=[],
    )
    set_up.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for set_up
    set_up.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    set_up.tStart = globalClock.getTime(format='float')
    set_up.status = STARTED
    thisExp.addData('set_up.started', set_up.tStart)
    set_up.maxDuration = None
    # keep track of which components have finished
    set_upComponents = set_up.components
    for thisComponent in set_up.components:
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
    
    # --- Run Routine "set_up" ---
    set_up.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
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
            set_up.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in set_up.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "set_up" ---
    for thisComponent in set_up.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for set_up
    set_up.tStop = globalClock.getTime(format='float')
    set_up.tStopRefresh = tThisFlipGlobal
    thisExp.addData('set_up.stopped', set_up.tStop)
    thisExp.nextEntry()
    # the Routine "set_up" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[instr_text, instr_leave],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_leave
    instr_leave.keys = []
    instr_leave.rt = []
    _instr_leave_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
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
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr_text* updates
        
        # if instr_text is starting this frame...
        if instr_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_text.frameNStart = frameN  # exact frame index
            instr_text.tStart = t  # local t and not account for scr refresh
            instr_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_text.started')
            # update status
            instr_text.status = STARTED
            instr_text.setAutoDraw(True)
        
        # if instr_text is active this frame...
        if instr_text.status == STARTED:
            # update params
            pass
        
        # *instr_leave* updates
        waitOnFlip = False
        
        # if instr_leave is starting this frame...
        if instr_leave.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_leave.frameNStart = frameN  # exact frame index
            instr_leave.tStart = t  # local t and not account for scr refresh
            instr_leave.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_leave, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_leave.started')
            # update status
            instr_leave.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_leave.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_leave.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_leave.status == STARTED and not waitOnFlip:
            theseKeys = instr_leave.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_leave_allKeys.extend(theseKeys)
            if len(_instr_leave_allKeys):
                instr_leave.keys = _instr_leave_allKeys[-1].name  # just the last key pressed
                instr_leave.rt = _instr_leave_allKeys[-1].rt
                instr_leave.duration = _instr_leave_allKeys[-1].duration
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
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if instr_leave.keys in ['', [], None]:  # No response was made
        instr_leave.keys = None
    thisExp.addData('instr_leave.keys',instr_leave.keys)
    if instr_leave.keys != None:  # we had a response
        thisExp.addData('instr_leave.rt', instr_leave.rt)
        thisExp.addData('instr_leave.duration', instr_leave.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    conditions_loop = data.TrialHandler2(
        name='conditions_loop',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(cond_file), 
        seed=None, 
    )
    thisExp.addLoop(conditions_loop)  # add the loop to the experiment
    thisConditions_loop = conditions_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisConditions_loop.rgb)
    if thisConditions_loop != None:
        for paramName in thisConditions_loop:
            globals()[paramName] = thisConditions_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisConditions_loop in conditions_loop:
        currentLoop = conditions_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisConditions_loop.rgb)
        if thisConditions_loop != None:
            for paramName in thisConditions_loop:
                globals()[paramName] = thisConditions_loop[paramName]
        
        # --- Prepare to start Routine "fix" ---
        # create an object to store info about Routine fix
        fix = data.Routine(
            name='fix',
            components=[fixation_cross],
        )
        fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for fix
        fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fix.tStart = globalClock.getTime(format='float')
        fix.status = STARTED
        thisExp.addData('fix.started', fix.tStart)
        fix.maxDuration = None
        # keep track of which components have finished
        fixComponents = fix.components
        for thisComponent in fix.components:
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
        
        # --- Run Routine "fix" ---
        # if trial has changed, end Routine now
        if isinstance(conditions_loop, data.TrialHandler2) and thisConditions_loop.thisN != conditions_loop.thisTrial.thisN:
            continueRoutine = False
        fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_cross* updates
            
            # if fixation_cross is starting this frame...
            if fixation_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross.frameNStart = frameN  # exact frame index
                fixation_cross.tStart = t  # local t and not account for scr refresh
                fixation_cross.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross.started')
                # update status
                fixation_cross.status = STARTED
                fixation_cross.setAutoDraw(True)
            
            # if fixation_cross is active this frame...
            if fixation_cross.status == STARTED:
                # update params
                pass
            
            # if fixation_cross is stopping this frame...
            if fixation_cross.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_cross.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_cross.tStop = t  # not accounting for scr refresh
                    fixation_cross.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_cross.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross.stopped')
                    # update status
                    fixation_cross.status = FINISHED
                    fixation_cross.setAutoDraw(False)
            
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
                fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fix" ---
        for thisComponent in fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fix
        fix.tStop = globalClock.getTime(format='float')
        fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fix.stopped', fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fix.maxDurationReached:
            routineTimer.addTime(-fix.maxDuration)
        elif fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "vas_indicator" ---
        # create an object to store info about Routine vas_indicator
        vas_indicator = data.Routine(
            name='vas_indicator',
            components=[slider_indicator],
        )
        vas_indicator.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from pain_indicator_slider
        slider_pain.marker = visual.Rect(win, width=0.01, height=0.1, lineColor='red', fillColor='red', units=slider_pain.units)
        
        for i, line in enumerate(slider_pain.tickLines.sizes):
            slider_pain.tickLines.sizes[i][1] = 0
        slider_pain.tickLines._needVertexUpdate = True
        
        #we will need to dynamically set the pos of slider using expected_vas
        if expected_vas == 30:
            pos = 30
        elif expected_vas == 70 :
            pos = 70
        
        slider_pain.startValue = pos
        slider_pain.reset()
        
        # Start a timer for max duration
        vas_timer = core.Clock()
        vas_timer.reset()
        slider_indicator.reset()
        # Run 'Begin Routine' code from boxes_creation
        #in this experiment we will use boxes to show uncertainty
        #In this part of the set_up we will start by creating these boxes so they can be called later
        
        
        percent = pos / 100  # convert to 0–1
        slider_width = slider_pain.size[0]
        slider_left = slider_pain.pos[0] - slider_width/2
        
        center = slider_left + percent * slider_width
        
        small_box = visual.Rect(
            win=win,
            width=slider_width * 0.05,       # 5% of slider width
            height=slider_pain.size[1],      # match slider height
            pos=(center, slider_pain.pos[1]),
            lineColor='white',
            fillColor='white',
            opacity=0.5,
            units=slider_pain.units
        )
        
        large_box = visual.Rect(
            win=win,
            width=slider_width * 0.35,        # 35% of slider width
            height=slider_pain.size[1],       # match slider height
            pos=(center, slider_pain.pos[1]),  # vertically aligned with slider
            lineColor='white',
            fillColor='white',
            opacity=0.5,                       # semi-transparent
            units=slider_pain.units
        )
        
        
        # store start times for vas_indicator
        vas_indicator.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        vas_indicator.tStart = globalClock.getTime(format='float')
        vas_indicator.status = STARTED
        thisExp.addData('vas_indicator.started', vas_indicator.tStart)
        vas_indicator.maxDuration = None
        # keep track of which components have finished
        vas_indicatorComponents = vas_indicator.components
        for thisComponent in vas_indicator.components:
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
        
        # --- Run Routine "vas_indicator" ---
        # if trial has changed, end Routine now
        if isinstance(conditions_loop, data.TrialHandler2) and thisConditions_loop.thisN != conditions_loop.thisTrial.thisN:
            continueRoutine = False
        vas_indicator.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from pain_indicator_slider
            
            inc = 0.75
            wasmoved = 0
            
            while True:
                time_elapsed = vas_timer.getTime()  # seconds since reset
                if time_elapsed >= 4.0:  # we want 4 seconds of visual representation for pain
                    rating = slider_pain.markerPos
                    core.wait(0.1)
                    vaspressed = 0
                    continueRoutine = False
                    break
                if expect_uncertainty == 'low':
                    small_box.draw()
            
                elif expect_uncertainty == 'high':
                    large_box.draw()
                
                # print(kb.state)
                slider_pain.draw()
                win.mouseVisible = False
                win.flip()
            
            # *slider_indicator* updates
            
            # if slider_indicator is starting this frame...
            if slider_indicator.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_indicator.frameNStart = frameN  # exact frame index
                slider_indicator.tStart = t  # local t and not account for scr refresh
                slider_indicator.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_indicator, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_indicator.started')
                # update status
                slider_indicator.status = STARTED
                slider_indicator.setAutoDraw(True)
            
            # if slider_indicator is active this frame...
            if slider_indicator.status == STARTED:
                # update params
                pass
            
            # Check slider_indicator for response to end Routine
            if slider_indicator.getRating() is not None and slider_indicator.status == STARTED:
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
                vas_indicator.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in vas_indicator.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "vas_indicator" ---
        for thisComponent in vas_indicator.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for vas_indicator
        vas_indicator.tStop = globalClock.getTime(format='float')
        vas_indicator.tStopRefresh = tThisFlipGlobal
        thisExp.addData('vas_indicator.stopped', vas_indicator.tStop)
        conditions_loop.addData('slider_indicator.response', slider_indicator.getRating())
        conditions_loop.addData('slider_indicator.rt', slider_indicator.getRT())
        # the Routine "vas_indicator" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pain_fix" ---
        # create an object to store info about Routine pain_fix
        pain_fix = data.Routine(
            name='pain_fix',
            components=[fake_pain],
        )
        pain_fix.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from pain_stim_code
        #we will need to add the pain stim code below, 
        #as of now this will be an empty code component
        #we will show a text with 'pain' to simulate and pilot
        
        # store start times for pain_fix
        pain_fix.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pain_fix.tStart = globalClock.getTime(format='float')
        pain_fix.status = STARTED
        thisExp.addData('pain_fix.started', pain_fix.tStart)
        pain_fix.maxDuration = None
        # keep track of which components have finished
        pain_fixComponents = pain_fix.components
        for thisComponent in pain_fix.components:
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
        
        # --- Run Routine "pain_fix" ---
        # if trial has changed, end Routine now
        if isinstance(conditions_loop, data.TrialHandler2) and thisConditions_loop.thisN != conditions_loop.thisTrial.thisN:
            continueRoutine = False
        pain_fix.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fake_pain* updates
            
            # if fake_pain is starting this frame...
            if fake_pain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fake_pain.frameNStart = frameN  # exact frame index
                fake_pain.tStart = t  # local t and not account for scr refresh
                fake_pain.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fake_pain, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fake_pain.started')
                # update status
                fake_pain.status = STARTED
                fake_pain.setAutoDraw(True)
            
            # if fake_pain is active this frame...
            if fake_pain.status == STARTED:
                # update params
                pass
            
            # if fake_pain is stopping this frame...
            if fake_pain.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fake_pain.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    fake_pain.tStop = t  # not accounting for scr refresh
                    fake_pain.tStopRefresh = tThisFlipGlobal  # on global time
                    fake_pain.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fake_pain.stopped')
                    # update status
                    fake_pain.status = FINISHED
                    fake_pain.setAutoDraw(False)
            
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
                pain_fix.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pain_fix.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pain_fix" ---
        for thisComponent in pain_fix.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pain_fix
        pain_fix.tStop = globalClock.getTime(format='float')
        pain_fix.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pain_fix.stopped', pain_fix.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if pain_fix.maxDurationReached:
            routineTimer.addTime(-pain_fix.maxDuration)
        elif pain_fix.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "vas_rating" ---
        # create an object to store info about Routine vas_rating
        vas_rating = data.Routine(
            name='vas_rating',
            components=[slider_pain, slider_pain_resp, main_prompt, space_prompt],
        )
        vas_rating.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from slider_pain_code
        
        
        slider_pain.marker = visual.Rect(win, width=0.01, height=0.1, lineColor='red', fillColor='red', units=slider_pain.units)
        
        for i, line in enumerate(slider_pain.tickLines.sizes):
            slider_pain.tickLines.sizes[i][1] = 0
        slider_pain.tickLines._needVertexUpdate = True
        pos = np.random.uniform(0, 100)
        slider_pain.reset()
        slider_pain.startValue = pos
        
        # Start a timer for max duration
        vas_timer = core.Clock()
        vas_timer.reset()
        slider_pain.reset()
        # create starting attributes for slider_pain_resp
        slider_pain_resp.keys = []
        slider_pain_resp.rt = []
        _slider_pain_resp_allKeys = []
        # store start times for vas_rating
        vas_rating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        vas_rating.tStart = globalClock.getTime(format='float')
        vas_rating.status = STARTED
        thisExp.addData('vas_rating.started', vas_rating.tStart)
        vas_rating.maxDuration = None
        # keep track of which components have finished
        vas_ratingComponents = vas_rating.components
        for thisComponent in vas_rating.components:
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
        
        # --- Run Routine "vas_rating" ---
        # if trial has changed, end Routine now
        if isinstance(conditions_loop, data.TrialHandler2) and thisConditions_loop.thisN != conditions_loop.thisTrial.thisN:
            continueRoutine = False
        vas_rating.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from slider_pain_code
            
            inc = 0.75
            wasmoved = 0
            
            while True:
                keys = slider_pain_resp.getKeys(['n', 'm', 'space', 'escape'], waitRelease=False, clear=False)
                if keys and not keys[-1].duration:
                    key = keys[-1].name
                    if 'n' == key:
                        slider_pain.markerPos -= inc
                        wasmoved = 1
                    if 'm' == key:
                        slider_pain.markerPos  += inc
                        wasmoved = 1
            
                    # check for quit (typically the Esc key)
                    if "escape" ==  key:
                        core.quit()
                        
                    if 'space' == key:
                        rating = slider_pain.markerPos
                        core.wait(0.1)
                        vaspressed = 1
                        thisExp.addData('VAS_rating', rating)
                        continueRoutine=False
                        break
                    time_elapsed = vas_timer.getTime()  # seconds since reset
                    if time_elapsed >= 10.0:  # after 10s, end routine we can change this to anything we want
                        rating = slider_pain.markerPos
                        core.wait(0.1)
                        vaspressed = 0
                        continueRoutine = False
                        break
            
            
                #print(kb.state)
                slider_pain.draw()
                main_prompt.draw()
                space_prompt.draw()
                win.mouseVisible = False
                win.flip()
            
            # *slider_pain* updates
            
            # if slider_pain is starting this frame...
            if slider_pain.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_pain.frameNStart = frameN  # exact frame index
                slider_pain.tStart = t  # local t and not account for scr refresh
                slider_pain.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_pain, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_pain.started')
                # update status
                slider_pain.status = STARTED
                slider_pain.setAutoDraw(True)
            
            # if slider_pain is active this frame...
            if slider_pain.status == STARTED:
                # update params
                pass
            
            # Check slider_pain for response to end Routine
            if slider_pain.getRating() is not None and slider_pain.status == STARTED:
                continueRoutine = False
            
            # *slider_pain_resp* updates
            waitOnFlip = False
            
            # if slider_pain_resp is starting this frame...
            if slider_pain_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_pain_resp.frameNStart = frameN  # exact frame index
                slider_pain_resp.tStart = t  # local t and not account for scr refresh
                slider_pain_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_pain_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_pain_resp.started')
                # update status
                slider_pain_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(slider_pain_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(slider_pain_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if slider_pain_resp.status == STARTED and not waitOnFlip:
                theseKeys = slider_pain_resp.getKeys(keyList=['n','m','enter', 'space'], ignoreKeys=["escape"], waitRelease=False)
                _slider_pain_resp_allKeys.extend(theseKeys)
                if len(_slider_pain_resp_allKeys):
                    slider_pain_resp.keys = _slider_pain_resp_allKeys[-1].name  # just the last key pressed
                    slider_pain_resp.rt = _slider_pain_resp_allKeys[-1].rt
                    slider_pain_resp.duration = _slider_pain_resp_allKeys[-1].duration
            
            # *main_prompt* updates
            
            # if main_prompt is starting this frame...
            if main_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                main_prompt.frameNStart = frameN  # exact frame index
                main_prompt.tStart = t  # local t and not account for scr refresh
                main_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(main_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'main_prompt.started')
                # update status
                main_prompt.status = STARTED
                main_prompt.setAutoDraw(True)
            
            # if main_prompt is active this frame...
            if main_prompt.status == STARTED:
                # update params
                pass
            
            # *space_prompt* updates
            
            # if space_prompt is starting this frame...
            if space_prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                space_prompt.frameNStart = frameN  # exact frame index
                space_prompt.tStart = t  # local t and not account for scr refresh
                space_prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(space_prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'space_prompt.started')
                # update status
                space_prompt.status = STARTED
                space_prompt.setAutoDraw(True)
            
            # if space_prompt is active this frame...
            if space_prompt.status == STARTED:
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
                vas_rating.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in vas_rating.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "vas_rating" ---
        for thisComponent in vas_rating.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for vas_rating
        vas_rating.tStop = globalClock.getTime(format='float')
        vas_rating.tStopRefresh = tThisFlipGlobal
        thisExp.addData('vas_rating.stopped', vas_rating.tStop)
        conditions_loop.addData('slider_pain.response', slider_pain.getRating())
        conditions_loop.addData('slider_pain.rt', slider_pain.getRT())
        # the Routine "vas_rating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'conditions_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_exp" ---
    # create an object to store info about Routine end_exp
    end_exp = data.Routine(
        name='end_exp',
        components=[end_exp_text, key_resp],
    )
    end_exp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for end_exp
    end_exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_exp.tStart = globalClock.getTime(format='float')
    end_exp.status = STARTED
    thisExp.addData('end_exp.started', end_exp.tStart)
    end_exp.maxDuration = None
    # keep track of which components have finished
    end_expComponents = end_exp.components
    for thisComponent in end_exp.components:
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
    
    # --- Run Routine "end_exp" ---
    end_exp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_exp_text* updates
        
        # if end_exp_text is starting this frame...
        if end_exp_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_exp_text.frameNStart = frameN  # exact frame index
            end_exp_text.tStart = t  # local t and not account for scr refresh
            end_exp_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_exp_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_exp_text.started')
            # update status
            end_exp_text.status = STARTED
            end_exp_text.setAutoDraw(True)
        
        # if end_exp_text is active this frame...
        if end_exp_text.status == STARTED:
            # update params
            pass
        
        # if end_exp_text is stopping this frame...
        if end_exp_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_exp_text.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                end_exp_text.tStop = t  # not accounting for scr refresh
                end_exp_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_exp_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_exp_text.stopped')
                # update status
                end_exp_text.status = FINISHED
                end_exp_text.setAutoDraw(False)
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
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
            end_exp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_exp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_exp" ---
    for thisComponent in end_exp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_exp
    end_exp.tStop = globalClock.getTime(format='float')
    end_exp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_exp.stopped', end_exp.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "end_exp" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
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
