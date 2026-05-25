function matlab_sender_server_liveframe_worldlock_change()
    clc;

    videoPort = 5000;
    metadataHost = "127.0.0.1";
    metadataPort = 5001;
    retargetLocalPort = 5002;

    fps = 5;
    jpegQuality = 40;
    runSeconds = 300;

    frameVarName = 'latestSimFrame';

    srcWidth = 1280;
    srcHeight = 720;

    fx = 1109;
    fy = 1109;
    cx = 640;
    cy = 360;

    defaultTargetWorld = [50, 0, 0];

    txScale = 0.5;

    cameraOffsetWorld = [0, 0, -20];

    modelName = 'uav_model';
    tgtCoordBlockPath = [modelName '/TGT Coordination'];

    targetMoveMode = "rate_limited";
    targetMoveRate = 60.0;
    targetExpAlpha = 0.25;

    tcpObj = tcpserver("0.0.0.0", videoPort);
    udpMeta = udpport("datagram", "IPV4");
    udpCmd  = udpport("datagram", "IPV4", "LocalPort", retargetLocalPort, "Timeout", 0.001);

    cleanupObj = onCleanup(@()localCleanup(tcpObj, udpMeta, udpCmd)); %#ok<NASGU>

    waitForClient(tcpObj);

    frameId = uint32(0);

    targetPixel = [round(srcWidth/2), round(srcHeight/2)];
    targetValid = false;

    desiredTargetWorld = defaultTargetWorld;
    smoothedTargetWorld = defaultTargetWorld;

    lastIntentAckKey = "";
    lastIntentCode = "";
    lastIntentTimestamp = 0;
    intentSeq = uint32(0);

    confirmPending = false;
    confirmTargetPixel = [NaN NaN];
    confirmTargetWorld = [NaN NaN NaN];

    assignDesiredAndSmoothedTargets(desiredTargetWorld, smoothedTargetWorld);
    applyTargetWorldToSimulink(modelName, tgtCoordBlockPath, smoothedTargetWorld);

    tStart = tic;
    nextTick = tic;

    while toc(tStart) < runSeconds
        frameId = frameId + 1;

        if ~isClientConnected(tcpObj)
            waitForClient(tcpObj);
            nextTick = tic;
        end

        ok = false;
        frame = [];

        try
            if evalin('base', sprintf('exist(''%s'',''var'')', frameVarName)) ~= 0
                frame = evalin('base', frameVarName);
                if ~isempty(frame)
                    ok = true;
                end
            end
        catch
            ok = false;
        end

        if ~ok
            pause(0.01);
            continue;
        end

        uavPos = finiteRow3(readBaseVarOrDefault('latestUavPos', [0 0 0]), [0 0 0]);
        camRPYDeg = finiteRow3(readBaseVarOrDefault('latestCamRPYDeg', [0 0 0]), [0 0 0]);

        [targetPixel, targetValid, action, lastIntentAckKey, lastIntentCode, lastIntentTimestamp, desiredTargetWorld, ...
         confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq] = ...
            pollRetarget(udpCmd, targetPixel, targetValid, srcWidth, srcHeight, ...
                         uavPos, camRPYDeg, fx, fy, cx, cy, cameraOffsetWorld, ...
                         defaultTargetWorld, desiredTargetWorld, ...
                         lastIntentAckKey, lastIntentCode, lastIntentTimestamp, ...
                         confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq);

        switch action.type
            case "retarget"
                [tmpPointCurrent, ~] = estimateClickedGroundPointCamWorld( ...
                    targetPixel, uavPos, camRPYDeg, ...
                    fx, fy, cx, cy, srcWidth, srcHeight, cameraOffsetWorld);

                if all(isfinite(tmpPointCurrent))
                    desiredTargetWorld = tmpPointCurrent;
                end

            case "cancel"
                desiredTargetWorld = defaultTargetWorld;
                confirmPending = false;
        end

        dt = 1 / fps;
        smoothedTargetWorld = stepTargetWorld( ...
            smoothedTargetWorld, desiredTargetWorld, dt, ...
            targetMoveMode, targetMoveRate, targetExpAlpha);

        assignDesiredAndSmoothedTargets(desiredTargetWorld, smoothedTargetWorld);
        applyTargetWorldToSimulink(modelName, tgtCoordBlockPath, smoothedTargetWorld);

        frame = normalizeFrameForJpeg(frame);

        if txScale ~= 1.0
            frame = imresize(frame, txScale);
        end

        try
            if isClientConnected(tcpObj)
                jpgBytes = encodeJpegToBytesStable(frame, jpegQuality);
                sendVideoFrame(tcpObj, jpgBytes);
            else
                continue;
            end
        catch
            pause(0.05);
            continue;
        end

        try
            payloadStr = buildMetadataJson( ...
                frameId, uavPos, desiredTargetWorld, targetPixel, targetValid, ...
                lastIntentAckKey, lastIntentCode, lastIntentTimestamp, confirmPending, intentSeq);

            payload = unicode2native(char(payloadStr), 'UTF-8');
            write(udpMeta, payload, "uint8", metadataHost, metadataPort);
        catch
        end

        targetPeriod = 1 / fps;
        elapsed = toc(nextTick);
        pauseTime = targetPeriod - elapsed;
        if pauseTime > 0
            pause(pauseTime);
        end
        nextTick = tic;
    end
end


function payloadStr = buildMetadataJson(frameId, uavPos, desiredTargetWorld, targetPixel, targetValid, intentAckKey, intentCode, intentTimestamp, confirmPending, intentSeq)
    if strlength(intentAckKey) == 0
        intentAckKey = "";
    end
    if strlength(intentCode) == 0
        intentCode = "";
    end

    safeAckKey = escapeJsonString(char(intentAckKey));
    safeCode = escapeJsonString(char(intentCode));

    payloadStr = sprintf([ ...
        '{"frame_id":%d,' ...
        '"uav_pos":[%.6f,%.6f,%.6f],' ...
        '"desired_target_world":[%.6f,%.6f,%.6f],' ...
        '"target_pixel":[%d,%d],' ...
        '"target_valid":%s,' ...
        '"intent_ack_key":"%s",' ...
        '"intent_code":"%s",' ...
        '"intent_timestamp":%.6f,' ...
        '"confirm_pending":%s,' ...
        '"intent_seq":%d}'], ...
        double(frameId), ...
        uavPos(1), uavPos(2), uavPos(3), ...
        desiredTargetWorld(1), desiredTargetWorld(2), desiredTargetWorld(3), ...
        round(targetPixel(1)), round(targetPixel(2)), ...
        lower(string(targetValid)), ...
        safeAckKey, safeCode, double(intentTimestamp), ...
        lower(string(confirmPending)), ...
        double(intentSeq));
end


function s = escapeJsonString(s)
    s = strrep(s, '\', '\\');
    s = strrep(s, '"', '\"');
end


function v = readBaseVarOrDefault(varName, defaultValue)
    try
        if evalin('base', sprintf('exist(''%s'',''var'')', varName))
            v = evalin('base', varName);
        else
            v = defaultValue;
        end
    catch
        v = defaultValue;
    end
end


function assignDesiredAndSmoothedTargets(desiredTargetWorld, smoothedTargetWorld)
    assignin('base', 'latestDesiredTargetWorld', desiredTargetWorld(:).');
    assignin('base', 'latestSmoothedTargetWorld', smoothedTargetWorld(:).');
end


function v = finiteRow3(v, fallback)
    v = forceRow3(v, fallback);
    if any(~isfinite(v))
        v = fallback;
    end
end


function v = forceRow3(v, defaultValue)
    try
        v = double(v(:).');
        if numel(v) < 3
            v = defaultValue;
        else
            v = v(1:3);
        end
    catch
        v = defaultValue;
    end
end


function [targetPixel, targetValid, action, intentAckKey, intentCode, intentTimestamp, desiredTargetWorld, ...
          confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq] = ...
    pollRetarget(udpCmd, targetPixel, targetValid, srcWidth, srcHeight, ...
                 uavPos, camRPYDeg, fx, fy, cx, cy, cameraOffsetWorld, ...
                 defaultTargetWorld, desiredTargetWorld, ...
                 intentAckKey, intentCode, intentTimestamp, ...
                 confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq)

    action = struct('type', "none");

    while udpCmd.NumDatagramsAvailable > 0
        try
            data = read(udpCmd, 1, "uint8");
            txt = char(data.Data(:))';
            cmd = jsondecode(txt);

            if ~isfield(cmd, "cmd")
                continue;
            end

            switch string(cmd.cmd)
                case "retarget"
                    if isfield(cmd, "src_pixel") && numel(cmd.src_pixel) == 2
                        x = round(double(cmd.src_pixel(1)));
                        y = round(double(cmd.src_pixel(2)));
                        x = max(1, min(srcWidth, x));
                        y = max(1, min(srcHeight, y));
                        targetPixel = [x, y];
                        targetValid = true;
                        action.type = "retarget";
                    end

                case "cancel_retarget"
                    targetValid = false;
                    action.type = "cancel";
                    confirmPending = false;

                case "multimodal_intent"
                    [targetPixel, targetValid, desiredTargetWorld, intentAckKey, intentCode, intentTimestamp, ...
                     confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq] = ...
                        handleMultimodalIntent(cmd, targetPixel, targetValid, desiredTargetWorld, ...
                                               uavPos, camRPYDeg, fx, fy, cx, cy, ...
                                               srcWidth, srcHeight, cameraOffsetWorld, ...
                                               defaultTargetWorld, confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq);
                    action.type = "multimodal";
            end
        catch
        end
    end
end


function [targetPixel, targetValid, desiredTargetWorld, intentAckKey, intentCode, intentTimestamp, ...
          confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq] = ...
    handleMultimodalIntent(cmd, targetPixel, targetValid, desiredTargetWorld, ...
                           uavPos, camRPYDeg, fx, fy, cx, cy, srcWidth, srcHeight, ...
                           cameraOffsetWorld, defaultTargetWorld, ...
                           confirmPending, confirmTargetPixel, confirmTargetWorld, intentSeq)

    intentAckKey = "";
    intentCode = "";
    intentTimestamp = now * 86400;

    fusedAction = "";
    voiceAction = "";
    gazeValid = false;
    gazePixel = [];

    try
        if isfield(cmd, "fusion") && isfield(cmd.fusion, "fused_action")
            fusedAction = string(cmd.fusion.fused_action);
        end

        if isfield(cmd, "voice") && isfield(cmd.voice, "voice_action")
            voiceAction = string(cmd.voice.voice_action);
        end

        if isfield(cmd, "gaze") && isfield(cmd.gaze, "valid")
            gazeValid = logical(cmd.gaze.valid);
        end

        if gazeValid && isfield(cmd, "gaze") && isfield(cmd.gaze, "src_pixel") && numel(cmd.gaze.src_pixel) == 2
            gx = round(double(cmd.gaze.src_pixel(1)));
            gy = round(double(cmd.gaze.src_pixel(2)));
            gx = max(1, min(srcWidth, gx));
            gy = max(1, min(srcHeight, gy));
            gazePixel = [gx, gy];
        end
    catch
    end

    if confirmPending
        if voiceAction == "attack"
            if all(isfinite(confirmTargetPixel))
                targetPixel = confirmTargetPixel;
                targetValid = true;
            end
            if all(isfinite(confirmTargetWorld))
                desiredTargetWorld = confirmTargetWorld;
            end

            intentAckKey = "ACK_ATTACK_NOW";
            intentCode = "confirm_resolved_attack";
            confirmPending = false;
            intentSeq = intentSeq + 1;
            return;

        else
            if all(isfinite(confirmTargetPixel))
                targetPixel = confirmTargetPixel;
                targetValid = true;
            end
            if all(isfinite(confirmTargetWorld))
                desiredTargetWorld = confirmTargetWorld;
            end

            intentAckKey = "ACK_TRACK_CURRENT";
            intentCode = "confirm_resolved_track";
            confirmPending = false;
            intentSeq = intentSeq + 1;
            return;
        end
    end

    switch fusedAction
        case "attack_current_target"
            intentAckKey = "ACK_ATTACK_NOW";
            intentCode = "attack_current_target";
            intentSeq = intentSeq + 1;

        case "track_current_target"
            intentAckKey = "ACK_TRACK_CURRENT";
            intentCode = "track_current_target";
            intentSeq = intentSeq + 1;

        case "hold_current_target"
            intentAckKey = "ACK_TRACK_CURRENT";
            intentCode = "hold_current_target";
            intentSeq = intentSeq + 1;

        case "attack_gaze_candidate"
            if ~isempty(gazePixel)
                targetPixel = gazePixel;
                targetValid = true;

                [pWorld, ~] = estimateClickedGroundPointCamWorld( ...
                    targetPixel, uavPos, camRPYDeg, ...
                    fx, fy, cx, cy, srcWidth, srcHeight, cameraOffsetWorld);

                if all(isfinite(pWorld))
                    desiredTargetWorld = pWorld;
                    confirmTargetWorld = pWorld;
                end
                confirmTargetPixel = targetPixel;
            else
                confirmTargetPixel = targetPixel;
                confirmTargetWorld = desiredTargetWorld;
            end

            confirmPending = true;
            intentAckKey = "ACK_CONFIRM_ATTACK";
            intentCode = "attack_gaze_candidate";
            intentSeq = intentSeq + 1;

        case "confirm_then_attack"
            if ~isempty(gazePixel)
                targetPixel = gazePixel;
                targetValid = true;

                [pWorld, ~] = estimateClickedGroundPointCamWorld( ...
                    targetPixel, uavPos, camRPYDeg, ...
                    fx, fy, cx, cy, srcWidth, srcHeight, cameraOffsetWorld);

                if all(isfinite(pWorld))
                    desiredTargetWorld = pWorld;
                    confirmTargetWorld = pWorld;
                end
                confirmTargetPixel = targetPixel;
            else
                confirmTargetPixel = targetPixel;
                confirmTargetWorld = desiredTargetWorld;
            end

            confirmPending = true;
            intentAckKey = "ACK_CONFIRM_ATTACK";
            intentCode = "confirm_then_attack";
            intentSeq = intentSeq + 1;

        case "track_gaze_candidate"
            if ~isempty(gazePixel)
                targetPixel = gazePixel;
                targetValid = true;

                [pWorld, ~] = estimateClickedGroundPointCamWorld( ...
                    targetPixel, uavPos, camRPYDeg, ...
                    fx, fy, cx, cy, srcWidth, srcHeight, cameraOffsetWorld);

                if all(isfinite(pWorld))
                    desiredTargetWorld = pWorld;
                end
            end
            intentAckKey = "ACK_TRACK_HERE";
            intentCode = "track_gaze_candidate";
            intentSeq = intentSeq + 1;

        otherwise
            % no new ack
    end

    if isempty(desiredTargetWorld) || any(~isfinite(desiredTargetWorld))
        desiredTargetWorld = defaultTargetWorld;
    end
end


function [worldPoint, reason] = estimateClickedGroundPointCamWorld( ...
    clickedPixel, uavPos, camRPYDeg, ...
    fx, fy, cx, cy, srcWidth, srcHeight, cameraOffsetWorld)

    reason = "ok";
    worldPoint = [NaN NaN NaN];

    if numel(clickedPixel) < 2 || numel(uavPos) < 3 || numel(camRPYDeg) < 3
        reason = "bad_input_size";
        return;
    end

    uavPos = double(uavPos(:).');
    camRPYDeg = double(camRPYDeg(:).');
    cameraOffsetWorld = double(cameraOffsetWorld(:).');

    if any(~isfinite(uavPos)) || any(~isfinite(camRPYDeg)) || any(~isfinite(cameraOffsetWorld))
        reason = "nonfinite_input";
        return;
    end

    u = max(1, min(srcWidth, double(clickedPixel(1))));
    v = max(1, min(srcHeight, double(clickedPixel(2))));

    if ~isfinite(fx) || ~isfinite(fy) || fx == 0 || fy == 0
        reason = "bad_intrinsics";
        return;
    end

    x_img = (u - cx) / fx;
    y_img = (v - cy) / fy;

    rayCam = [1; -x_img; -y_img];
    rayCam = rayCam / norm(rayCam);

    R_world_from_cam = rpyDegToRotmZYX(camRPYDeg(1), camRPYDeg(2), camRPYDeg(3));
    rayWorld = R_world_from_cam * rayCam;
    rayWorld = rayWorld / norm(rayWorld);

    cameraPosWorld = uavPos + cameraOffsetWorld;

    if cameraPosWorld(3) <= 0 || abs(rayWorld(3)) < 1e-9
        reason = "bad_ray";
        return;
    end

    t = (0 - cameraPosWorld(3)) / rayWorld(3);
    if t <= 0
        reason = "intersection_behind_camera";
        return;
    end

    p = cameraPosWorld(:) + t * rayWorld;
    worldPoint = p(:).';
end


function R = rpyDegToRotmZYX(rollDeg, pitchDeg, yawDeg)
    cr = cosd(rollDeg);  sr = sind(rollDeg);
    cp = cosd(pitchDeg); sp = sind(pitchDeg);
    cy = cosd(yawDeg);   sy = sind(yawDeg);

    Rx = [1 0 0; 0 cr -sr; 0 sr cr];
    Ry = [cp 0 sp; 0 1 0; -sp 0 cp];
    Rz = [cy -sy 0; sy cy 0; 0 0 1];

    R = Rz * Ry * Rx;
end


function nextTarget = stepTargetWorld(currentTarget, desiredTarget, dt, moveMode, moveRate, expAlpha)
    currentTarget = double(currentTarget(:).');
    desiredTarget = double(desiredTarget(:).');

    if any(~isfinite(currentTarget)) || numel(currentTarget) < 3
        nextTarget = desiredTarget;
        return;
    end

    if any(~isfinite(desiredTarget)) || numel(desiredTarget) < 3
        nextTarget = currentTarget;
        return;
    end

    switch string(moveMode)
        case "exp"
            a = max(0.0, min(1.0, expAlpha));
            nextTarget = currentTarget + a * (desiredTarget - currentTarget);

        otherwise
            maxStep = max(0, moveRate * dt);
            delta = desiredTarget - currentTarget;
            dist = norm(delta);

            if dist <= maxStep || dist < 1e-9
                nextTarget = desiredTarget;
            else
                nextTarget = currentTarget + (maxStep / dist) * delta;
            end
    end
end


function applyTargetWorldToSimulink(modelName, blockPath, worldPoint)
    if numel(worldPoint) < 3 || any(~isfinite(worldPoint))
        return;
    end

    worldPoint = double(worldPoint(:).');

    try
        if ~bdIsLoaded(modelName)
            load_system(modelName);
        end

        valueStr = sprintf('[%.15g; %.15g; %.15g]', ...
            worldPoint(1), worldPoint(2), worldPoint(3));

        set_param(blockPath, 'Value', valueStr);
    catch
    end
end


function frame = normalizeFrameForJpeg(frame)
    if isa(frame, 'double') || isa(frame, 'single')
        if max(frame(:)) <= 1.0
            frame = uint8(255 * frame);
        else
            frame = uint8(frame);
        end
    end

    if ndims(frame) == 2
        frame = repmat(frame, [1 1 3]);
    end

    if size(frame, 3) == 1
        frame = repmat(frame, [1 1 3]);
    elseif size(frame, 3) > 3
        frame = frame(:, :, 1:3);
    end

    frame = uint8(frame);
end


function jpgBytes = encodeJpegToBytesStable(frame, jpegQuality)
    persistent fixedFilePath
    if isempty(fixedFilePath)
        fixedFilePath = fullfile(tempdir, 'frame_buffer.jpg');
    end

    imwrite(frame, fixedFilePath, "jpg", "Quality", jpegQuality);

    fid = fopen(fixedFilePath, 'rb');
    if fid < 0
        error('Failed to open JPEG file for reading.');
    end

    cleaner = onCleanup(@() fclose(fid)); %#ok<NASGU>
    jpgBytes = fread(fid, inf, '*uint8');

    if isempty(jpgBytes)
        error('JPEG byte buffer is empty.');
    end
end


function sendVideoFrame(tcpObj, jpgBytes)
    n = numel(jpgBytes);
    header = typecast(swapbytes(uint32(n)), 'uint8');
    write(tcpObj, header, "uint8");
    write(tcpObj, jpgBytes, "uint8");
end


function waitForClient(tcpObj)
    while ~isClientConnected(tcpObj)
        pause(0.1);
    end
end


function tf = isClientConnected(tcpObj)
    tf = false;
    try
        tf = logical(tcpObj.Connected);
    catch
        tf = false;
    end
end


function localCleanup(tcpObj, udpMeta, udpCmd)
    try, clear tcpObj; catch, end
    try, clear udpMeta; catch, end
    try, clear udpCmd; catch, end
end