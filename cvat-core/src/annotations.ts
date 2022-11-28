// Copyright (C) 2019-2022 Intel Corporation
// Copyright (C) 2022 CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { Storage } from './storage';

const serverProxy = require('./server-proxy').default;
const Collection = require('./annotations-collection');
const AnnotationsSaver = require('./annotations-saver');
const AnnotationsHistory = require('./annotations-history').default;
const { checkObjectType } = require('./common');
const Project = require('./project').default;
const { Task, Job } = require('./session');
const { ScriptingError, DataError, ArgumentError } = require('./exceptions');
const { getDeletedFrames } = require('./frames');

const jobCache = new WeakMap();
const taskCache = new WeakMap();

function getCache(sessionType) {
    if (sessionType === 'task') {
        return taskCache;
    }

    if (sessionType === 'job') {
        return jobCache;
    }

    throw new ScriptingError(`Unknown session type was received ${sessionType}`);
}

async function getAnnotationsFromServer(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (!cache.has(session)) {
        const rawAnnotations = await serverProxy.annotations.getAnnotations(sessionType, session.id);

        // Get meta information about frames
        const startFrame = sessionType === 'job' ? session.startFrame : 0;
        const stopFrame = sessionType === 'job' ? session.stopFrame : session.size - 1;
        const frameMeta = {};
        for (let i = startFrame; i <= stopFrame; i++) {
            frameMeta[i] = await session.frames.get(i);
        }
        frameMeta.deleted_frames = await getDeletedFrames(sessionType, session.id);

        const history = new AnnotationsHistory();
        const collection = new Collection({
            labels: session.labels || session.task.labels,
            history,
            startFrame,
            stopFrame,
            frameMeta,
        });

        // eslint-disable-next-line no-unsanitized/method
        collection.import(rawAnnotations);
        const saver = new AnnotationsSaver(rawAnnotations.version, collection, session);
        cache.set(session, { collection, saver, history });
    }
}

export async function clearCache(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        cache.delete(session);
    }
}

export async function getAnnotations(session, frame, allTracks, filters) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.get(frame, allTracks, filters);
    }

    await getAnnotationsFromServer(session);
    return cache.get(session).collection.get(frame, allTracks, filters);
}

export async function saveAnnotations(session, onUpdate) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        await cache.get(session).saver.save(onUpdate);
    }

    // If a collection wasn't uploaded, than it wasn't changed, finally we shouldn't save it
}

export function searchAnnotations(session, filters, frameFrom, frameTo) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.search(filters, frameFrom, frameTo);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function searchEmptyFrame(session, frameFrom, frameTo) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.searchEmpty(frameFrom, frameTo);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function mergeAnnotations(session, objectStates) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.merge(objectStates);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function splitAnnotations(session, objectState, frame) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.split(objectState, frame);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function groupAnnotations(session, objectStates, reset) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.group(objectStates, reset);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function hasUnsavedChanges(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).saver.hasUnsavedChanges();
    }

    return false;
}

export async function clearAnnotations(session, reload, startframe, endframe, delTrackKeyframesOnly) {
    checkObjectType('reload', reload, 'boolean', null);
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        cache.get(session).collection.clear(startframe, endframe, delTrackKeyframesOnly);
    }

    if (reload) {
        cache.delete(session);
        await getAnnotationsFromServer(session);
    }
}

export function annotationsStatistics(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.statistics();
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function putAnnotations(session, objectStates) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.put(objectStates);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function selectObject(session, objectStates, x, y) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.select(objectStates, x, y);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function importCollection(session, data) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        // eslint-disable-next-line no-unsanitized/method
        return cache.get(session).collection.import(data);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function exportCollection(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).collection.export();
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export async function exportDataset(
    instance,
    format: string,
    saveImages: boolean,
    useDefaultSettings: boolean,
    targetStorage: Storage,
    name?: string,
) {
    if (!(instance instanceof Task || instance instanceof Project || instance instanceof Job)) {
        throw new ArgumentError('A dataset can only be created from a job, task or project');
    }

    let result = null;
    if (instance instanceof Task) {
        result = await serverProxy.tasks
            .exportDataset(instance.id, format, saveImages, useDefaultSettings, targetStorage, name);
    } else if (instance instanceof Job) {
        result = await serverProxy.jobs
            .exportDataset(instance.id, format, saveImages, useDefaultSettings, targetStorage, name);
    } else {
        result = await serverProxy.projects
            .exportDataset(instance.id, format, saveImages, useDefaultSettings, targetStorage, name);
    }

    return result;
}

export function importDataset(
    instance: any,
    format: string,
    useDefaultSettings: boolean,
    sourceStorage: Storage,
    file: File | string,
    options: {
        convMaskToPoly?: boolean,
        updateStatusCallback?: (s: string, n: number) => void,
    } = {},
): Promise<void> {
    const updateStatusCallback = options.updateStatusCallback || (() => {});
    const convMaskToPoly = 'convMaskToPoly' in options ? options.convMaskToPoly : true;
    const adjustedOptions = {
        updateStatusCallback,
        convMaskToPoly,
    };

    if (!(instance instanceof Project || instance instanceof Task || instance instanceof Job)) {
        throw new ArgumentError('Instance must be a Project || Task || Job instance');
    }
    if (!(typeof updateStatusCallback === 'function')) {
        throw new ArgumentError('Callback must be a function');
    }
    if (!(typeof convMaskToPoly === 'boolean')) {
        throw new ArgumentError('Option "convMaskToPoly" must be a boolean');
    }
    if (typeof file === 'string' && !file.toLowerCase().endsWith('.zip')) {
        throw new ArgumentError('File must be file instance with ZIP extension');
    }
    if (file instanceof File && !(['application/zip', 'application/x-zip-compressed'].includes(file.type))) {
        throw new ArgumentError('File must be file instance with ZIP extension');
    }

    if (instance instanceof Project) {
        return serverProxy.projects
            .importDataset(
                instance.id,
                format,
                useDefaultSettings,
                sourceStorage,
                file,
                adjustedOptions,
            );
    }

    const instanceType = instance instanceof Task ? 'task' : 'job';
    return serverProxy.annotations
        .uploadAnnotations(
            instanceType,
            instance.id,
            format,
            useDefaultSettings,
            sourceStorage,
            file,
            adjustedOptions,
        );
}

export function getHistory(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).history;
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export async function undoActions(session, count) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).history.undo(count);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export async function redoActions(session, count) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).history.redo(count);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function freezeHistory(session, frozen) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).history.freeze(frozen);
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function clearActions(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).history.clear();
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}

export function getActions(session) {
    const sessionType = session instanceof Task ? 'task' : 'job';
    const cache = getCache(sessionType);

    if (cache.has(session)) {
        return cache.get(session).history.get();
    }

    throw new DataError(
        'Collection has not been initialized yet. Call annotations.get() or annotations.clear(true) before',
    );
}
