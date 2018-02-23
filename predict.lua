require 'paths'
json = require 'json'
paths.dofile('util.lua')
paths.dofile('img.lua')


--------------------------------------------------------------------------------
-- Utility functions
--------------------------------------------------------------------------------


function add_pose_data (clip)
  local clip = clip

  -- Process image if no pose data yet
  if not has_array(clip, 'points_2d') then
    print('Processing clip ' .. clip['id'])

    images = image_names(clip['id'], clip['start'], clip['end'])
    clip['points_2d'] = predict_poses(clip, images)
  end

  return clip
end

function predict_poses (clip, images)
  xlua.progress(0, #images)

  local points = {}

  if not has(clip, 'scale') then
    print("ERROR: No scale parameter present")
    return nil
  end
  local scale = clip['scale']

  if not has_array(clip, 'center') then
    print("ERROR: No center point present")
    return nil
  end
  local center = clip['center']

  for i = 1, #images do
    -- Load image
    local img = image.load(images[i])

    -- Normalize and center
    local input = crop(img, center, scale, 0, 256)

    -- Predict pose
    local out = m:forward(input:view(1,3,256,256):cuda())
    cutorch.synchronize()
    local hm = out[#out][1]:float()
    hm[hm:lt(0)] = 0
    local preds_hm, preds_img = getPreds(hm, center, scale)

    -- Store result
    points[#points + 1] = torch.totable(preds_img:view(preds_img:nElement()))

    -- Continue
    xlua.progress(i, #images)
    collectgarbage()
  end

  return points
end

function image_names (id, start_frame, end_frame)
  local images = {}
  start_frame = math.floor(start_frame)
  end_frame = math.floor(end_frame)

  for j = 1, (end_frame - start_frame) do
    images[j] = image_root .. id .. '-' .. tostring(j) .. image_extension
  end

  return images
end

function read_file (file_name)
  local file = assert(io.open(file_name, 'rb'))
  local content = file:read('*all')
  file:close()
  return content
end

function has (tbl, index)
  return tbl[index] ~= nil
end

function has_array (tbl, index)
  return (has(tbl, index) and #tbl[index] ~= 0)
end


--------------------------------------------------------------------------------
-- Main code
--------------------------------------------------------------------------------


if #arg < 1 then
  print("No config file given")
  return
end

config = json.decode(read_file(arg[1]))

-- Read configuration
image_root = config['image_root']
image_extension = config['image_extension']
print("Loading images from " .. image_root .. " with extension " .. image_extension)

m = torch.load('umich-stacked-hourglass.t7')   -- Load pre-trained model

-- Add pose data
for i = 1, #config['clips'] do
  local clip = config['clips'][i]
  config['clips'][i] = add_pose_data(clip)
end

-- Write new config file
config_file = io.open(arg[1], 'w')
if config_file then
  config_file:write(json.encode(config))
  config_file:close()
  return 0
else
  print("Could not open config file for writing")
  return 1
end
