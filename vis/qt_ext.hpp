#ifdef QIMAGE_H

#ifndef TIPL_QT_EXT_HPP
#define TIPL_QT_EXT_HPP

#include <QGraphicsView>
#include <QScrollBar>


// show image on scene and keep the original scroll bar position if zoom in/out
inline void operator<<(QGraphicsScene& scene,QImage I)
{
    float vb_ratio(0),hb_ratio(0);
    QScrollBar* vb(nullptr),*hb(nullptr);
    auto views = scene.views();
    if(!views.empty() && int(float(scene.sceneRect().width())/float(scene.sceneRect().height())*100.0f) ==
           int(float(I.width())/float(I.height())*100.0f))
    {
        vb = views[0]->verticalScrollBar();
        hb = views[0]->horizontalScrollBar();
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
   return image = QImage(reinterpret_cast<const unsigned char*>(I.data()),I.width(),I.height(),QImage::Format_RGB32).copy();
}

inline QImage operator << (QImage&&,const tipl::color_image& I)
{
   return QImage(reinterpret_cast<const unsigned char*>(I.data()),I.width(),I.height(),QImage::Format_RGB32).copy();
}

template<typename value_type>
inline tipl::image<2,value_type>& operator << (tipl::image<2,value_type>& image,const QImage& I)
{
    QImage I2 = I.convertToFormat(QImage::Format_RGB32);
    const uchar* ptr = I2.bits();
    const size_t total_size = I2.width()*I2.height();
    image.resize(tipl::shape<2>(uint32_t(I2.width()),uint32_t(I2.height())));
    for(size_t j = 0;j < total_size;++j,ptr += 4)
        image[j] = value_type(tipl::rgb(*(ptr+2),*(ptr+1),*ptr));
    return image;
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
                bool grid = false,
                float tic_ratio = 1.0f)
{
    tipl::vector<3> qsdr_scale(trans[0],trans[5],trans[10]);
    tipl::vector<3> qsdr_shift(trans[3],trans[7],trans[11]);

    float zoom_2 = zoom*0.5f;

    int tic = 50;
    float tic_dis = 10.0f; // in mm
    if(std::fabs(qsdr_scale[0])*5.0f/zoom < 1.0f)
    {
        tic = 10;
        tic_dis = 5.0f; // in mm
    }
    if(std::fabs(qsdr_scale[0])*5.0f/zoom < 0.4f)
    {
        tic = 10;
        tic_dis = 2.0f;
    }
    if(std::fabs(qsdr_scale[0])*5.0f/zoom < 0.2f)
    {
        tic = 5;
        tic_dis = 1.0f;
    }
    if(tic_ratio != 1.0f)
    {
        tic *= tic_ratio;
        tic_dis *= tic_ratio;
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

    const size_t sz_h = tic_pos_h.size();
    {
        auto Y = max_Y+zoom_2;
        paint.drawLine(int(min_X-zoom_2),int(Y),int(max_X+zoom_2),int(Y));
        for(size_t i = 0;i < sz_h;++i)
        {
            bool is_tic = !(int(tic_value_h[i]) % tic);
            auto X = tic_pos_h[i]+zoom_2;
            if(is_tic)
            {
                paint.setPen(pen2);
                paint.drawLine(int(X),int(grid ? min_Y+zoom_2 : Y),int(X),int(Y+zoom));
            }
            paint.setPen(pen1);
            paint.drawLine(int(X),int(grid ? min_Y+zoom_2 : Y),int(X),int(Y+zoom));
            paint.setFont(is_tic ? f2:f1);
            paint.drawText(int(X-40),int(Y-30+zoom),80,80,
                           Qt::AlignHCenter|Qt::AlignVCenter,QString::number(tic_value_h[i]));
        }
    }

    const size_t sz_v = tic_pos_v.size();
    {
        auto X = min_X+zoom_2;
        paint.drawLine(int(X),int(min_Y-zoom_2),int(X),int(max_Y+zoom_2));
        for(size_t i = 0;i < sz_v;++i)
        {
            bool is_tic = !(int(tic_value_v[i]) % tic);
            auto Y = tic_pos_v[i]+zoom_2;
            if(is_tic)
            {
                paint.setPen(pen2);
                paint.drawLine(int(grid ? max_X : X),int(Y),int(X-zoom),int(Y));
            }
            paint.setPen(pen1);
            paint.drawLine(int(grid ? max_X : X),int(Y),int(X-zoom),int(Y));
            paint.setFont(is_tic ? f2:f1);
            paint.drawText(2,int(Y-40),int(X-zoom)-5,80,
                           Qt::AlignRight|Qt::AlignVCenter,QString::number(tic_value_v[i]));
        }
    }
}

template<typename image_type>
QImage draw_regions(const std::vector<image_type>& region_masks,
                  const std::vector<tipl::rgb>& colors,
                  bool draw_edge,int line_width,
                  int cur_roi_index,
                  float display_ratio)
{
    if(region_masks.empty())
        return QImage();
    auto dim = region_masks[0].shape();
    const int w = dim.width();
    const int h = dim.height();
    const size_t dim_size = dim.size(); // HOISTED to avoid repeating W*H evaluation
    const size_t rm_size = region_masks.size();

    // draw region colors on the image
    tipl::color_image slice_image_with_region(dim);  //original slices for adding regions pixels
    // draw regions and also derive where the edges are
    std::vector<std::vector<tipl::vector<2,int> > > edge_x(rm_size), edge_y(rm_size);
    {
        tipl::par_for(rm_size,[&](uint32_t roi_index)
        {
            auto& region_mask = region_masks[roi_index];
            auto color = colors[roi_index];
            bool draw_roi = (!draw_edge && color.a >= 128);
            // detect edge
            auto& cur_edge_x = edge_x[roi_index];
            auto& cur_edge_y = edge_y[roi_index];

            for(tipl::pixel_index<2> index(dim); index < dim_size; ++index)
            {
                size_t idx = index.index();
                if(region_mask[idx])
                {
                    auto x = index[0];
                    auto y = index[1];
                    if(draw_roi)
                        slice_image_with_region[idx] = color;
                    if(y > 0 && !region_mask[idx-w])
                        cur_edge_x.push_back(tipl::vector<2,int>(x,y));
                    if(y+1 < h && !region_mask[idx+w])
                        cur_edge_x.push_back(tipl::vector<2,int>(x,y+1));
                    if(x > 0 && !region_mask[idx-1])
                        cur_edge_y.push_back(tipl::vector<2,int>(x,y));
                    if(x+1 < w && !region_mask[idx+1])
                        cur_edge_y.push_back(tipl::vector<2,int>(x+1,y));
                }
            }
        });
    }

    // now apply image scaling to the slice image
    QImage scaled_image = (QImage() << slice_image_with_region).scaled(int(w*display_ratio),int(h*display_ratio));
    if(draw_edge)
    {
        unsigned int foreground_color = ((scaled_image.pixel(0,0) & 0x000000FF) < 128 ? 0xFFFFFFFF:0xFF000000);
        QPainter paint(&scaled_image);
        for (uint32_t roi_index = 0; roi_index < rm_size; ++roi_index)
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
    const int w = p.width();
    const int h = p.height();
    int l = w, r = 0, t = h, b = 0;

    auto first_pixel = p.pixel(0,0);
    for (int y = 0; y < h; ++y) {
        auto row = reinterpret_cast<decltype(&first_pixel)>(p.scanLine(y));
        bool rowFilled = false;
        for (int x = 0; x < w; ++x)
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
    r = std::min(w-1,r+margin);
    t = std::max(0,t-margin);
    b = std::min(h-1,b+margin);
    return p.copy(QRect(l,t,r-l,b-t));
}

inline QImage create_mosaic(const std::vector<QImage>& images,int col_size)
{
    const size_t num_images = images.size();
    if(num_images == 0) return QImage();

    int height = 0,width = 0;
    for (size_t i = 0; i < num_images; ++i)
    {
        height = std::max<int>(images[i].height(),height);
        width = std::max<int>(images[i].width(),width);
    }
    width += 5;
    height += 5;

    QImage I(num_images >= static_cast<size_t>(col_size) ? width*col_size : width*int(num_images),
             height*int(1+num_images/col_size),QImage::Format_RGB32);
    I.fill(images[0].pixel(0,0));

    QPainter painter(&I);
    painter.setCompositionMode(QPainter::CompositionMode_Source);

    for (size_t i = 0, j = 0; i < num_images; ++i, ++j)
        painter.drawImage(int(j%col_size)*width+(width-images[i].width())/2,
                          int(i/col_size)*height+(height-images[i].height())/2,images[i]);
    return I;
}

}//qt
}//tipl

#endif//TIPL_QT_EXT_HPP


#ifdef TIPL_GZ_STREAM_HPP

#ifndef TIPL_QT_EXT2_HPP
#define TIPL_QT_EXT2_HPP

#include <QFileDialog>
#include <QFileIconProvider>
#include <QGridLayout>
#include <QLabel>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPointer>
#include <QFileInfo>
#include <QLineEdit>
#include <QSortFilterProxyModel>
#include <QFileSystemModel>
#include <QSlider>
#include <QPushButton>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QDir>
#include <QWheelEvent>
#include <QMouseEvent>
#include <atomic>
#include <thread>
#include <sstream>
#include "../io/gz_stream.hpp"
#include "../io/nifti.hpp"
#include "../io/dicom.hpp"
#include "../io/mat.hpp"
#include "../io/2dseq.hpp"



namespace tipl::qt
{

inline QList<QUrl> working_dirs;

namespace details
{

inline bool read_preview_image(QString file,tipl::image<3,float>& I,
                              tipl::image<3,unsigned char>& dseg,
                              std::string& report)
{
    tipl::vector<3> vs;
    tipl::matrix<4,4> T;
    auto fn = file.toUtf8();
    auto finish = [&]()
    {
        std::ostringstream out;
        out << "image size: [" << I.width() << "," << I.height() << "," << I.depth()
            << "] x (" << vs[0] << " mm," << vs[1] << " mm," << vs[2] << " mm)\n"
            << "srow:\n" << std::fixed << std::setprecision(4);
        for(int r = 0;r < 3;++r)
            out << std::setw(10) << T[r*4] << " " << std::setw(10) << T[r*4+1] << " "
                << std::setw(10) << T[r*4+2] << " " << std::setw(10) << T[r*4+3] << "\n";
        report = out.str() + report;
        return !I.empty();
    };

    if(file.endsWith("nii.gz") || file.endsWith("nii"))
    {
        std::scoped_lock<std::mutex> lock(tipl::io::nifti_do_not_show_process);
        if(tipl::io::gz_nifti in(fn.constData(),std::ios::in);in >> I >> vs >> T)
        {
            std::ostringstream out;
            out << in;
            report = out.str();

            QFileInfo info(file);
            auto name = info.fileName();
            if(name.startsWith("sub-") && name.endsWith(".nii.gz") && !name.endsWith("_dseg.nii.gz"))
            {
                name.chop(7);
                int p = name.lastIndexOf('_');
                if(p != -1)
                {
                    tipl::image<3,unsigned char> D;
                    tipl::vector<3> dvs;
                    tipl::matrix<4,4> DT;
                    tipl::shape<3> Ddim;
                    auto dseg_file = info.absolutePath() + "/" + name.left(p) + "_dseg.nii.gz";
                    auto dseg_fn = dseg_file.toUtf8();
                    if(tipl::io::gz_nifti din(dseg_fn.constData(),std::ios::in);
                       din >> Ddim >> dvs >> DT && Ddim == I.shape() && dvs == vs && DT == T)
                        {
                            din >> D;
                            dseg.swap(D);
                        }
                }
            }
            return finish();
        }
    }
    else
        if(file.endsWith("2dseq"))
        {
            tipl::io::bruker_2dseq seq;
            if(seq.load_from_file(fn.constData()))
                seq.get_image().swap(I),seq.get_voxel_size(vs);
        }
        else
            if(file.endsWith(".dcm"))
            {
                tipl::io::dicom dicom;
                if(dicom.load_from_file(fn.constData()))
                    dicom >> std::tie(I,vs,report);
            }
            else
                if(file.endsWith("z"))
                {
                    tipl::shape<3> dim;
                    if(tipl::io::gz_mat_read in;in.load_from_file(fn.constData()) &&
                       in.read_pointer("dimension",dim) && in.read_pointer("voxel_size",vs))
                    {
                        in.read_pointer("trans",T);
                        if(const unsigned char* mask = nullptr;in.read("mask",mask))
                            in.si2vi = tipl::get_sparse_index(tipl::make_image(mask,dim)),
                                in.mask_cols = dim.plane_size(),in.mask_rows = dim.depth();

                        for(auto each : {"iso","image0","subject0"})
                            if(in.has(each))
                            {
                                I.resize(dim);
                                in.read(each,I);
                                break;
                            }
                        report = in.read<std::string>("report");
                    }
                }
    return finish();
}

template<typename T>
void add_preview(T& dlg)
{
    struct preview_label : QLabel{
        QDoubleSpinBox* zoom = nullptr;
        QSlider* slider = nullptr;
        QCheckBox* edge = nullptr;
        std::function<void()> redraw;
        int x = 0,y = 0;
        float wl = 0.0f,ww = 1.0f;
        bool fixed_wl = false;
        using QLabel::QLabel;
        void mouseDoubleClickEvent(QMouseEvent*) override    {edge->setChecked(!edge->isChecked());redraw();}
        void wheelEvent(QWheelEvent* e) override             {zoom->setValue(zoom->value() + (e->angleDelta().y() > 0 ? zoom->singleStep() : -zoom->singleStep()));}
        void mousePressEvent(QMouseEvent* e) override
        {
            x = int(e->position().x());
            y = int(e->position().y());
        }
        void mouseMoveEvent(QMouseEvent* e) override
        {
            int x2 = int(e->position().x()),y2 = int(e->position().y());

            if(e->buttons() & Qt::LeftButton)
            {
                if(int d = (y-y2)/4)
                    slider->setValue(slider->value()+d),y -= d*4;
                x = x2;
                return;
            }

            int dx = x2-x,dy = y2-y;
            if(e->buttons() & Qt::RightButton)
                fixed_wl = true,ww = std::max(1.0f,ww+dx),wl -= dy,redraw();
            x = x2;
            y = y2;
        }
    };
    auto* preview = new preview_label("preview");
    auto* report = new QTextEdit;
    auto* panel = new QWidget;
    auto* box = new QVBoxLayout(panel);
    auto* ctrl = new QHBoxLayout;
    auto* slider = new QSlider(Qt::Horizontal);
    auto* sag = new QPushButton("Sag");
    auto* cor = new QPushButton("Cor");
    auto* axi = new QPushButton("Axi");
    auto* edge = new QCheckBox("dseg");
    auto* zoom = new QDoubleSpinBox;

    zoom->setRange(0.5,8.0);
    zoom->setSingleStep(0.5);
    zoom->setValue(2.0);

    preview->zoom = zoom;
    preview->slider = slider;
    preview->edge = edge;

    edge->setChecked(true);

    preview->setFixedSize(512,512);
    preview->setAlignment(Qt::AlignCenter);
    preview->setStyleSheet("background:#222;color:white");
    report->setReadOnly(true);
    report->setMinimumHeight(120);
    report->setStyleSheet("QTextEdit{background:#222;color:white;}");

    ctrl->addWidget(edge);
    ctrl->addWidget(zoom);
    for(auto* b : {sag,cor,axi})
    {
        b->setCheckable(true);
        ctrl->addWidget(b);
    }
    axi->setChecked(true);

    slider->setRange(0,2);
    slider->setValue(1);

    ctrl->insertWidget(0,slider,1);

    box->setContentsMargins(0,0,0,0);
    box->addWidget(preview);
    box->addLayout(ctrl);
    box->addWidget(report,1);

    if(auto* g = qobject_cast<QGridLayout*>(dlg.layout()))
        g->addWidget(panel,0,3,g->rowCount(),1);

    auto id = std::make_shared<std::atomic<size_t>>(0);
    auto I = std::make_shared<tipl::image<3,float>>();
    auto dseg = std::make_shared<tipl::image<3,unsigned char>>();
    auto dim = std::make_shared<int>(2);
    QPointer<QFileDialog> dlg_ptr(&dlg);

    auto redraw = [=]()
    {
        if(I->empty())
            return;

        auto slice = tipl::volume2slice_scaled(*I,*dim,size_t(slider->value()),float(zoom->value()));
        if(*dim != 2)
            tipl::flip_y(slice);
        auto [mn,mx] = std::minmax_element(slice.begin(),slice.end());
        if(!preview->fixed_wl)
            preview->wl = (*mn+*mx)*0.5f,preview->ww = std::max<float>(*mx-*mn,1.0f);
        float low = preview->wl-preview->ww*0.5f,scale = 255.0f/preview->ww;

        const int w = slice.width(),h = slice.height();
        QImage out(w,h,QImage::Format_RGB32);
        for(int y = 0;y < h;++y)
        {
            auto* dst = reinterpret_cast<QRgb*>(out.scanLine(y));
            auto* src = slice.data() + size_t(y)*w;
            for(int x = 0;x < w;++x,++src)
            {
                auto v = uchar(std::clamp<int>((*src-low)*scale,0,255));
                dst[x] = qRgb(v,v,v);
            }
        }

        if(edge->isChecked() && !dseg->empty())
        {
            auto label = tipl::volume2slice(*dseg,*dim,size_t(slider->value()));
            if(*dim != 2)
                tipl::flip_y(label);

            std::array<int,256> id;
            id.fill(-1);
            std::vector<tipl::image<2,uint8_t>> masks;
            std::vector<tipl::rgb> colors;

            for(size_t j = 0;j < label.size();++j)
                if(label[j])
                {
                    auto& cur = id[label[j]];
                    if(cur == -1)
                    {
                        cur = int(masks.size());
                        masks.emplace_back(label.shape());
                        colors.push_back(tipl::rgb::generate_hue(label[j]));
                        colors.back().a = 255;
                    }
                    masks[cur][j] = 1;
                }

            if(!masks.empty())
            {
                QPainter painter(&out);
                painter.setCompositionMode(QPainter::CompositionMode_SourceAtop);
                painter.drawImage(0,0,tipl::qt::draw_regions(masks,colors,1,1,-1,float(zoom->value())));
            }
        }
        preview->setPixmap(QPixmap::fromImage(out));
    };

    preview->redraw = redraw;

    auto set_dim = [=](int d)
    {
        float pos = slider->maximum() ? float(slider->value())/float(slider->maximum()) : 0.5f;
        int max_pos = I->empty() ? 0 : std::max<int>(0,int(I->shape()[d])-1);

        *dim = d;
        sag->setChecked(d == 0);
        cor->setChecked(d == 1);
        axi->setChecked(d == 2);

        slider->blockSignals(true);
        slider->setRange(0,max_pos);
        slider->setValue(int(pos*max_pos + 0.5f));
        slider->blockSignals(false);

        redraw();
    };

    QObject::connect(slider,&QSlider::valueChanged,redraw);
    QObject::connect(edge,&QCheckBox::clicked,redraw);
    QObject::connect(zoom,qOverload<double>(&QDoubleSpinBox::valueChanged),redraw);
    QObject::connect(sag,&QPushButton::clicked,[=]{set_dim(0);});
    QObject::connect(cor,&QPushButton::clicked,[=]{set_dim(1);});
    QObject::connect(axi,&QPushButton::clicked,[=]{set_dim(2);});

    QObject::connect(&dlg,&QFileDialog::currentChanged,[=](const QString& file)
    {
        size_t cur_id = ++(*id);
        preview->setText("loading...");
        preview->setPixmap({});
        report->clear();

        std::thread([=]
        {
            auto new_I = std::make_shared<tipl::image<3,float>>();
            auto new_dseg = std::make_shared<tipl::image<3,unsigned char>>();
            std::string r;
            read_preview_image(file,*new_I,*new_dseg,r);
            QMetaObject::invokeMethod(dlg_ptr,[=]
            {
                if(cur_id != *id)
                    return;
                if(new_I->empty())
                {
                    preview->setText("no preview");
                    return;
                }

                I->swap(*new_I);
                dseg->swap(*new_dseg);
                report->setPlainText(QString::fromStdString(r));
                preview->fixed_wl = false;
                set_dim(*dim);
            },Qt::QueuedConnection);
        }).detach();
    });
}

class simple_icon_provider : public QFileIconProvider
{
public:
    QIcon icon(const QFileInfo& info) const override
    {
        return QFileIconProvider::icon(info.isDir() ? Folder : File);
    }
};

class filename_filter_proxy : public QSortFilterProxyModel
{
    QStringList groups;
public:
    using QSortFilterProxyModel::QSortFilterProxyModel;

    void set_filter(QString text)
    {
        groups = text.split('|',Qt::SkipEmptyParts);
        invalidateFilter();
    }

protected:
    bool filterAcceptsRow(int row,const QModelIndex& parent) const override
    {
        if(groups.empty() || !sourceModel())
            return true;

        auto index = sourceModel()->index(row,0,parent);
        auto* fs = qobject_cast<QFileSystemModel*>(sourceModel());
        if(fs && fs->isDir(index))
            return true;

        QString name = fs ? fs->fileName(index) : sourceModel()->data(index).toString();

        for(const auto& group : groups)
        {
            bool ok = true;
            for(const auto& key : group.split(' ',Qt::SkipEmptyParts))
                if(!name.contains(key,Qt::CaseInsensitive))
                {
                    ok = false;
                    break;
                }
            if(ok)
                return true;
        }
        return false;
    }
};

template<typename T>
auto image_dialog(T* parent,QString path,QString filter,QFileDialog::AcceptMode accept,QFileDialog::FileMode mode,QString suffix = {})
{
    QString dir = QDir::currentPath(),name;
    if(!path.isEmpty())
    {
        QFileInfo info(path);
        if(info.isDir())
            dir = info.absoluteFilePath();
        else
        {
            dir = info.absolutePath();
            name = info.fileName();
            if(dir.isEmpty() || dir == ".")
                dir = QDir::currentPath();
        }
    }

    QFileDialog dlg(parent,accept == QFileDialog::AcceptSave ? "Save Image" : "Open Image");
    dlg.setOption(QFileDialog::DontUseNativeDialog,true);
    dlg.setOption(QFileDialog::DontUseCustomDirectoryIcons,true);
    dlg.setOption(QFileDialog::DontResolveSymlinks,true);
    dlg.setAcceptMode(accept);
    dlg.setFileMode(mode);
    dlg.setNameFilter(filter);
    dlg.setSidebarUrls(tipl::qt::working_dirs);
    dlg.setDirectory(dir);
    dlg.setIconProvider(new simple_icon_provider);

    if(accept == QFileDialog::AcceptOpen)
    {
        auto* proxy = new filename_filter_proxy(&dlg);
        auto* name_filter = new QLineEdit(&dlg);
        name_filter->setPlaceholderText("filter filename...");
        name_filter->setClearButtonEnabled(true);

        dlg.setProxyModel(proxy);
        QObject::connect(name_filter,&QLineEdit::textChanged,
                         proxy,&filename_filter_proxy::set_filter);

        if(auto* g = qobject_cast<QGridLayout*>(dlg.layout()))
        {
            int row = g->rowCount();
            g->addWidget(new QLabel("Filename contains:",&dlg),row,0);
            g->addWidget(name_filter,row,1,1,2);
        }
    }

    dlg.resize(1500,600);

    if(!suffix.isEmpty())
        dlg.setDefaultSuffix(suffix);
    if(!name.isEmpty())
        dlg.selectFile(name);
    if(accept == QFileDialog::AcceptOpen)
        add_preview(dlg);

    return dlg.exec() ? dlg.selectedFiles() : QStringList();
}

}

template<typename T>
auto open_image_files(T* parent,QString path,QString filter)
{
    return tipl::qt::details::image_dialog(parent,path,filter,QFileDialog::AcceptOpen,QFileDialog::ExistingFiles);
}

template<typename T>
auto open_image_file(T* parent,QString path,QString filter)
{
    auto files = tipl::qt::details::image_dialog(parent,path,filter,QFileDialog::AcceptOpen,QFileDialog::ExistingFile);
    return files.isEmpty() ? QString() : files.front();
}

template<typename T>
auto save_image_file(T* parent,QString default_name,QString filter)
{
    auto files = tipl::qt::details::image_dialog(parent,default_name,filter,QFileDialog::AcceptSave,QFileDialog::AnyFile,filter);
    return files.isEmpty() ? QString() : files.front();
}

inline std::filesystem::path to_path(const QString& s)
{
#ifdef _WIN32
    return std::filesystem::path(s.toStdWString());
#else
    return std::filesystem::path(s.toUtf8().constData());
#endif
}
inline QString to_qstring(const std::filesystem::path& p)
{
#ifdef _WIN32
    return QString::fromStdWString(p.wstring());
#else
#if defined(__cpp_char8_t)
    auto s = p.u8string();
    return QString::fromUtf8(reinterpret_cast<const char*>(s.data()),qsizetype(s.size()));
#else
    return QString::fromUtf8(p.u8string().c_str());
#endif
#endif
}

}


#endif//TIPL_QT_EXT2_HPP

#endif//TIPL_GZ_STREAM_HPP


#endif//QIMAGE_H
