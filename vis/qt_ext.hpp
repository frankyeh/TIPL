#ifdef QGRAPHICSSCENE_H

#ifndef TIPL_QT_EXT_HPP
#define TIPL_QT_EXT_HPP

#include <QGraphicsView>
#include <QScrollBar>



// show image on scene and keep the original scroll bar position if zoom in/out
inline void operator<<(QGraphicsScene& scene,QImage I)
{
    float vb_ratio(0),hb_ratio(0);
    QScrollBar* vb(nullptr),*hb(nullptr);
    if(scene.views().size() && int(float(scene.sceneRect().width())/float(scene.sceneRect().height())*100.0f) ==
           int(float(I.width())/float(I.height())*100.0f))
    {
        vb = scene.views()[0]->verticalScrollBar();
        hb = scene.views()[0]->horizontalScrollBar();
        if(vb->isVisible())
            vb_ratio = float((vb->value()+vb->pageStep()/2))/float(vb->maximum()+vb->pageStep());
        if(hb->isVisible())
            hb_ratio = float((hb->value()+hb->pageStep()/2))/float(hb->maximum()+hb->pageStep());
    }
    {
        scene.setSceneRect(0, 0, I.width(),I.height());
        scene.clear();
#ifdef WIN32
        scene.addPixmap(QPixmap::fromImage(I));
#else
        //For Mac, the endian system is BGRA and all QImage needs to be converted.
        scene.addPixmap(QPixmap::fromImage(I.convertToFormat(QImage::Format_ARGB32)));
#endif
    }

    if(vb_ratio != 0.0f)
        vb->setValue(int(vb_ratio*(vb->maximum()+vb->pageStep())-vb->pageStep()/2));
    if(hb_ratio != 0.0f)
        hb->setValue(int(hb_ratio*(hb->maximum()+hb->pageStep())-hb->pageStep()/2));
}
inline QImage& operator << (QImage& image,const tipl::color_image& I)
{
   return image = QImage((unsigned char*)&*I.begin(),
                  I.width(),I.height(),QImage::Format_RGB32).copy();

}
inline QImage operator << (QImage&&,const tipl::color_image& I)
{
   return QImage((unsigned char*)&*I.begin(),
                  I.width(),I.height(),QImage::Format_RGB32).copy();

}


namespace tipl{
namespace qt{

inline QPixmap image2pixelmap(const QImage &I)
{
    #ifdef WIN32
        return QPixmap::fromImage(I);
    #else
        //For Mac, the endian system is BGRA and all QImage needs to be converted.
        return QPixmap::fromImage(I.convertToFormat(QImage::Format_ARGB32));
    #endif
}



inline void draw_ruler(QPainter& paint,
                const tipl::shape<3>& shape,
                const tipl::matrix<4,4>& trans,
                unsigned char cur_dim,
                bool flip_x,bool flip_y,
                float zoom,
                bool grid = false)
{



    tipl::vector<3> qsdr_scale(trans[0],trans[5],trans[10]);
    tipl::vector<3> qsdr_shift(trans[3],trans[7],trans[11]);

    float zoom_2 = zoom*0.5f;

    int tick = 50;
    float tic_dis = 10.0f; // in mm
    if(std::fabs(qsdr_scale[0])*5.0f/zoom < 1.0f)
    {
        tick = 10;
        tic_dis = 5.0f; // in mm
    }
    if(std::fabs(qsdr_scale[0])*5.0f/zoom < 0.4f)
    {
        tick = 10;
        tic_dis = 2.0f;
    }
    if(std::fabs(qsdr_scale[0])*5.0f/zoom < 0.2f)
    {
        tick = 5;
        tic_dis = 1.0f;
    }

    float tic_length = zoom*float(shape[0])/20.0f;

    auto pen1 = paint.pen();  // creates a default pen
    auto pen2 = paint.pen();  // creates a default pen
    pen1.setColor(QColor(0xFF, 0xFF, 0xFF, 0xB0));
    pen1.setWidth(std::max<int>(1,int(zoom_2)));
    pen1.setCapStyle(Qt::RoundCap);
    pen1.setJoinStyle(Qt::RoundJoin);
    pen2.setColor(QColor(0xFF, 0xFF, 0xFF, 0x70));
    pen2.setWidth(std::max<int>(1,int(zoom)));
    pen2.setCapStyle(Qt::RoundCap);
    pen2.setJoinStyle(Qt::RoundJoin);
    paint.setPen(pen1);
    auto f1 = paint.font();
    f1.setPointSize(std::max<int>(1,zoom*tic_dis/std::abs(qsdr_scale[0])/3.5f));
    auto f2 = f1;
    f2.setBold(true);
    paint.setFont(f1);


    std::vector<float> tic_pos_h,tic_pos_v;
    std::vector<float> tic_value_h,tic_value_v;

    uint8_t dim_h = (cur_dim == 0 ? 1:0);
    uint8_t dim_v = (cur_dim == 2 ? 1:2);


    auto get_tic_pos = [zoom,tic_dis](std::vector<float>& tic_pos,
                          std::vector<float>& tic_value,
                          unsigned int shape_length,
                          float shift,float scale,
                          float margin1,float margin2,
                          bool flip)
         {
             float window_length = zoom*float(shape_length);
             float from = shift;
             float to = float(shape_length)*scale+from;
             if(from > to)
                 std::swap(from,to);
             from = std::floor(from/tic_dis)*tic_dis;
             to = std::ceil(to/tic_dis)*tic_dis;
             for(float pos = from;pos < to;pos += tic_dis)
             {
                 float pos_in_voxel = float(pos-shift)/scale;
                 pos_in_voxel = zoom*float(flip ? float(shape_length)-pos_in_voxel-1 : pos_in_voxel);
                 if(pos_in_voxel < margin1 || pos_in_voxel + margin2 > window_length)
                     continue;
                 tic_pos.push_back(pos_in_voxel);
                 tic_value.push_back(pos);
             }
         };

    get_tic_pos(tic_pos_h,tic_value_h,
                shape[dim_h],qsdr_shift[dim_h],qsdr_scale[dim_h],
                tic_length,tic_length,flip_x);
    get_tic_pos(tic_pos_v,tic_value_v,
                shape[dim_v],qsdr_shift[dim_v],qsdr_scale[dim_v],
                cur_dim == 0 ? tic_length/2:tic_length,tic_length,flip_y);

    if(tic_pos_h.empty() || tic_pos_v.empty())
        return;
    auto min_Y = std::min(tic_pos_v.front(),tic_pos_v.back());
    auto max_Y = std::max(tic_pos_v.front(),tic_pos_v.back());
    auto min_X = std::min(tic_pos_h.front(),tic_pos_h.back());
    auto max_X = std::max(tic_pos_h.front(),tic_pos_h.back());

    {
        auto Y = max_Y+zoom_2;
        paint.drawLine(int(min_X-zoom_2),int(Y),int(max_X+zoom_2),int(Y));
        for(size_t i = 0;i < tic_pos_h.size();++i)
        {
            bool is_tick = !(int(tic_value_h[i]) % tick);
            auto X = tic_pos_h[i]+zoom_2;
            if(is_tick)
            {
                paint.setPen(pen2);
                paint.drawLine(int(X),int(grid ? min_Y+zoom_2 : Y),int(X),int(Y+zoom));
            }
            paint.setPen(pen1);
            paint.drawLine(int(X),int(grid ? min_Y+zoom_2 : Y),int(X),int(Y+zoom));
            paint.setFont(is_tick ? f2:f1);
            paint.drawText(int(X-40),int(Y-30+zoom),80,80,
                           Qt::AlignHCenter|Qt::AlignVCenter,QString::number(tic_value_h[i]));
        }
    }
    {

        auto X = min_X+zoom_2;
        paint.drawLine(int(X),int(min_Y-zoom_2),int(X),int(max_Y+zoom_2));
        for(size_t i = 0;i < tic_pos_v.size();++i)
        {
            bool is_tick = !(int(tic_value_v[i]) % tick);
            auto Y = tic_pos_v[i]+zoom_2;
            if(is_tick)
            {
                paint.setPen(pen2);
                paint.drawLine(int(grid ? max_X : X),int(Y),int(X-zoom),int(Y));
            }
            paint.setPen(pen1);
            paint.drawLine(int(grid ? max_X : X),int(Y),int(X-zoom),int(Y));
            paint.setFont(is_tick ? f2:f1);
            paint.drawText(2,int(Y-40),int(X-zoom)-5,80,
                           Qt::AlignRight|Qt::AlignVCenter,QString::number(tic_value_v[i]));
        }
    }
}


template<typename image_type>
QImage draw_regions(const std::vector<image_type>& region_masks,
                  const std::vector<tipl::rgb>& colors,
                  bool fill_region,bool draw_edge,int line_width,
                  int cur_roi_index,
                  float display_ratio)
{
    if(region_masks.empty())
        return QImage();
    auto dim = region_masks[0].shape();
    int w = dim.width();
    int h = dim.height();
    // draw region colors on the image
    tipl::color_image slice_image_with_region(dim);  //original slices for adding regions pixels
    // draw regions and also derive where the edges are
    std::vector<std::vector<tipl::vector<2,int> > > edge_x(region_masks.size()),
                                                    edge_y(region_masks.size());
    {
        tipl::par_for(region_masks.size(),[&](uint32_t roi_index)
        {
            auto& region_mask = region_masks[roi_index];
            auto color = colors[roi_index];
            bool draw_roi = (fill_region && color.a >= 128);
            // detect edge
            auto& cur_edge_x = edge_x[roi_index];
            auto& cur_edge_y = edge_y[roi_index];
            for(tipl::pixel_index<2> index(dim);index < dim.size();++index)
                if(region_mask[index.index()])
                {
                    auto x = index[0];
                    auto y = index[1];
                    if(draw_roi)
                        slice_image_with_region[index.index()] = color;
                    if(y > 0 && !region_mask[index.index()-w])
                        cur_edge_x.push_back(tipl::vector<2,int>(x,y));
                    if(y+1 < h &&!region_mask[index.index()+w])
                        cur_edge_x.push_back(tipl::vector<2,int>(x,y+1));
                    if(x > 0 && !region_mask[index.index()-1])
                        cur_edge_y.push_back(tipl::vector<2,int>(x,y));
                    if(x+1 < w && !region_mask[index.index()+1])
                        cur_edge_y.push_back(tipl::vector<2,int>(x+1,y));
                }
        });
    }
    // now apply image scaling to the slice image
    QImage scaled_image = (QImage() << slice_image_with_region).scaled(int(w*display_ratio),int(h*display_ratio));
    if(draw_edge)
    {
        unsigned int foreground_color = ((scaled_image.pixel(0,0) & 0x000000FF) < 128 ? 0xFFFFFFFF:0xFF000000);
        QPainter paint(&scaled_image);
        for (uint32_t roi_index = 0;roi_index < region_masks.size();++roi_index)
        {
            unsigned int cur_color = foreground_color;
            if(int(roi_index) != cur_roi_index)
                cur_color = colors[roi_index];
            paint.setBrush(Qt::NoBrush);
            QPen pen(QColor(cur_color),line_width, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin);
            paint.setPen(pen);
            for(auto& pos : edge_x[roi_index])
            {
                pos *= display_ratio;
                paint.drawLine(pos[0],pos[1],pos[0]+display_ratio,pos[1]);
            }
            for(auto& pos : edge_y[roi_index])
            {
                pos *= display_ratio;
                paint.drawLine(pos[0],pos[1],pos[0],pos[1]+display_ratio);
            }
        }
    }
    return scaled_image;
}

template<typename image_type>
inline image_type get_bounding_box(image_type p,int margin = 5)
{
    int l =p.width(), r = 0, t = p.height(), b = 0;
    auto first_pixel = p.pixel(0,0);
    for (int y = 0; y < p.height(); ++y) {
        auto row = reinterpret_cast<decltype(&first_pixel)>(p.scanLine(y));
        bool rowFilled = false;
        for (int x = 0; x < p.width(); ++x)
        {
            if (row[x] != first_pixel)
            {
                rowFilled = true;
                r = std::max(r, x);
                if (l > x) {
                    l = x;
                    x = r; // shortcut to only search for new right bound from here
                }
            }
        }
        if (rowFilled) {
            t = std::min(t, y);
            b = y;
        }
    }
    l = std::max(0,l-margin);
    r = std::min(p.width()-1,r+margin);
    t = std::max(0,t-margin);
    b = std::min(p.height()-1,b+margin);
    return p.copy(QRect(l,t,r-l,b-t));
}

inline QImage create_mosaic(const std::vector<QImage>& images,int col_size)
{
    int height = 0,width = 0;
    for (auto& I : images)
        {
            height = std::max<int>(I.height(),height);
            width = std::max<int>(I.width(),width);
        }
    width += 5;
    height += 5;
    QImage I(images.size() >= col_size ? width*int(col_size): width*int(images.size()),
             height*int(1+images.size()/col_size),QImage::Format_RGB32);
    I.fill(images[0].pixel(0,0));
    QPainter painter(&I);
    painter.setCompositionMode(QPainter::CompositionMode_Source);
    for (size_t i = 0,j = 0;i < images.size();++i,++j)
        painter.drawImage(int(j%col_size)*width+(width-images[i].width())/2,
                          int(i/col_size)*height+(height-images[i].height())/2,images[i]);
    return I;
}

}//qt
}//tipl



#endif//TIPL_QT_EXT_HPP
#endif//QGRAPHICSSCENE_H
