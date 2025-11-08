-- Script để thêm nhiều đồng hồ mẫu cho testing
-- Chạy script này để có đủ dữ liệu test recommendations

-- Thêm đồng hồ mẫu
INSERT INTO watches (
    name, description, base_price, category_id, brand_id, 
    gender, status, del_flag, created_at, created_by,
    -- ML features
    price_tier, gender_target, size_category,
    style_tags, material_tags, color_tags, movement_type_tags,
    rating, sold
) VALUES 
-- Luxury watches
('Rolex Submariner Date', 'Classic diving watch', 85000000, 1, 1, 'M', true, '0', '20231201120000', 1, 'luxury', 'M', 'large', '["diving", "sport", "classic"]', '["steel", "ceramic"]', '["black", "blue"]', '["automatic"]', 4.8, 150),
('Omega Speedmaster', 'Moon landing watch', 45000000, 2, 2, 'M', true, '0', '20231201120000', 1, 'premium', 'M', 'medium', '["chronograph", "space", "vintage"]', '["steel", "leather"]', '["black", "white"]', '["manual"]', 4.7, 200),
('TAG Heuer Carrera', 'Racing chronograph', 25000000, 2, 3, 'M', true, '0', '20231201120000', 1, 'mid', 'M', 'medium', '["racing", "sport", "modern"]', '["steel", "rubber"]', '["black", "red"]', '["automatic"]', 4.5, 180),

-- Women's watches
('Cartier Tank', 'Elegant rectangular watch', 35000000, 3, 4, 'F', true, '0', '20231201120000', 1, 'premium', 'F', 'small', '["dress", "elegant", "classic"]', '["steel", "leather"]', '["silver", "black"]', '["quartz"]', 4.6, 120),
('Chanel J12', 'Ceramic luxury watch', 40000000, 3, 5, 'F', true, '0', '20231201120000', 1, 'luxury', 'F', 'medium', '["luxury", "modern", "elegant"]', '["ceramic", "steel"]', '["white", "black"]', '["automatic"]', 4.4, 90),

-- Budget watches
('Seiko SKX007', 'Affordable diver', 8000000, 1, 6, 'M', true, '0', '20231201120000', 1, 'budget', 'M', 'large', '["diving", "sport", "affordable"]', '["steel", "rubber"]', '["black", "blue"]', '["automatic"]', 4.3, 300),
('Citizen Eco-Drive', 'Solar powered watch', 12000000, 4, 7, 'U', true, '0', '20231201120000', 1, 'budget', 'U', 'medium', '["eco", "sport", "casual"]', '["steel", "nylon"]', '["black", "green"]', '["quartz"]', 4.2, 250),

-- Mid-range watches
('Tissot PRX', 'Swiss automatic', 18000000, 4, 8, 'M', true, '0', '20231201120000', 1, 'mid', 'M', 'medium', '["sport", "modern", "swiss"]', '["steel", "leather"]', '["blue", "black"]', '["automatic"]', 4.4, 220),
('Hamilton Khaki', 'Field watch', 15000000, 5, 9, 'M', true, '0', '20231201120000', 1, 'mid', 'M', 'medium', '["military", "field", "vintage"]', '["steel", "canvas"]', '["green", "brown"]', '["automatic"]', 4.3, 190),

-- Luxury dress watches
('Patek Philippe Calatrava', 'Ultra luxury dress', 120000000, 3, 10, 'M', true, '0', '20231201120000', 1, 'luxury', 'M', 'medium', '["dress", "luxury", "classic"]', '["gold", "leather"]', '["gold", "black"]', '["manual"]', 4.9, 50),
('Audemars Piguet Royal Oak', 'Luxury sports', 150000000, 2, 11, 'M', true, '0', '20231201120000', 1, 'luxury', 'M', 'large', '["luxury", "sport", "iconic"]', '["steel", "gold"]', '["steel", "blue"]', '["automatic"]', 4.8, 80);

-- Kiểm tra số lượng đồng hồ sau khi thêm
SELECT COUNT(*) as total_watches FROM watches WHERE del_flag = '0';

-- Kiểm tra các đồng hồ có sẵn
SELECT id, name, base_price, rating FROM watches WHERE del_flag = '0' ORDER BY id;
