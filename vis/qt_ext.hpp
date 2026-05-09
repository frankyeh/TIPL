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
#include <QGridLayout>
#include <QLabel>
#include <QTextEdit>
#include <QVBoxLayout>
#include <QPointer>
#include <QFileInfo>
#include <QDir>
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

inline bool read_preview_image(QString file,tipl::image<3,float>& I,tipl::vector<3>& vs,std::string& report)
{
    if(file.endsWith("nii.gz") || file.endsWith("nii"))
    {
        std::scoped_lock<std::mutex> lock(tipl::io::nifti_do_not_show_process);
        if(tipl::io::gz_nifti in(file.toStdString(),std::ios::in);in >> I >> vs)
        {
            std::ostringstream out;
            out << in;
            report = "\n" + out.str();
            return true;
        }
    }
    else
    if(file.endsWith("2dseq"))
    {
        tipl::io::bruker_2dseq seq;
        if(seq.load_from_file(file.toStdString()))
        {
            seq.get_image().swap(I);
            seq.get_voxel_size(vs);
            return true;
        }
    }
    else
    if(file.endsWith(".dcm"))
    {
        tipl::io::dicom dicom;
        if(dicom.load_from_file(file.toStdString()))
        {
            dicom >> std::tie(I,vs,report);
            return true;
        }
    }
    else
    if(file.endsWith("z"))
    {
        tipl::shape<3> dim;
        if(tipl::io::gz_mat_read in;
            in.load_from_file(file.toStdString()) &&
            in.read_pointer("dimension",dim) &&
            in.read_pointer("voxel_size",vs))
        {
            if(const unsigned char* mask_ptr = nullptr;in.read("mask",mask_ptr))
            {
                in.si2vi = tipl::get_sparse_index(tipl::make_image(mask_ptr,dim));
                in.mask_cols = dim.plane_size();
                in.mask_rows = dim.depth();
            }

            for(auto each : {"iso","image0","subject0"})
                if(in.has(each))
                {
                    I.resize(dim);
                    in.read(each,I);
                    break;
                }

            report = in.read<std::string>("report");
            return !I.empty();
        }
    }
    return false;
}

template<typename T>
void add_preview(T& dlg)
{
    auto* preview = new QLabel("preview");
    auto* report = new QTextEdit;
    auto* panel = new QWidget;
    auto* box = new QVBoxLayout(panel);

    preview->setFixedSize(512,512);
    preview->setAlignment(Qt::AlignCenter);
    preview->setStyleSheet("background:#222;color:white");
    report->setReadOnly(true);
    report->setMinimumHeight(120);
    report->setStyleSheet("QTextEdit{background:#222;color:white;}");

    box->setContentsMargins(0,0,0,0);
    box->addWidget(preview);
    box->addWidget(report,1);

    if(auto* g = qobject_cast<QGridLayout*>(dlg.layout()))
        g->addWidget(panel,0,3,g->rowCount(),1);

    auto* id = new std::atomic<size_t>(0);
    QPointer<QLabel> preview_ptr(preview);
    QPointer<QTextEdit> report_ptr(report);
    QPointer<QFileDialog> dlg_ptr(&dlg);




    QObject::connect(&dlg,&QFileDialog::destroyed,[id]{delete id;});

    QObject::connect(&dlg,&QFileDialog::currentChanged,
                     [id,preview_ptr,report_ptr,dlg_ptr](const QString& file)
                     {
                         size_t cur_id = ++(*id);
                         preview_ptr->setText("loading...");
                         preview_ptr->setPixmap({});
                         report_ptr->clear();

                         std::thread([file,cur_id,id,preview_ptr,report_ptr,dlg_ptr]
                                     {
                                         auto image_preview = [](const tipl::image<3,float>& I)
                                         {
                                             if(I.empty())
                                                 return QImage();

                                             int w = I.width(),h = I.height(),z = I.depth()/2;
                                             size_t off = size_t(z)*I.plane_size();
                                             auto [mn,mx] = std::minmax_element(I.begin()+off,I.begin()+off+I.plane_size());
                                             float d = *mx-*mn; if(d == 0.0f) d = 1.0f;
                                             float scale = 255.0f/d;

                                             QImage out(w,h,QImage::Format_Grayscale8);
                                             for(int y = 0;y < h;++y)
                                             {
                                                 auto* p = out.scanLine(y);
                                                 size_t pos = off + size_t(y)*w;
                                                 for(int x = 0;x < w;++x)
                                                     p[x] = uchar(std::clamp<int>((I[pos+x]-*mn)*scale,0,255));
                                             }
                                             return out.scaled(512,512,Qt::KeepAspectRatio,Qt::SmoothTransformation);
                                         };

                                         tipl::image<3,float> I;
                                         tipl::vector<3> vs;
                                         std::string r;
                                         read_preview_image(file,I,vs,r);

                                         QImage img;
                                         QString text;
                                         if(!I.empty())
                                         {
                                             img = image_preview(I);
                                             text = QString("image size: [%1,%2,%3] x (%4 mm,%5 mm,%6 mm)\n")
                                                        .arg(I.width()).arg(I.height()).arg(I.depth())
                                                        .arg(vs[0]).arg(vs[1]).arg(vs[2]);

                                             if(!r.empty())
                                                 text += QString::fromStdString("report:" + r);
                                         }

                                         if(!dlg_ptr)
                                             return;

                                         QMetaObject::invokeMethod(dlg_ptr,[=,img = std::move(img),text = std::move(text)]
                                             {
                                                 if(cur_id != *id || !preview_ptr || !report_ptr)
                                                     return;

                                                 if(img.isNull())
                                                 {
                                                     preview_ptr->setText("no preview");
                                                     preview_ptr->setPixmap({});
                                                     report_ptr->clear();
                                                     return;
                                                 }

                                                 preview_ptr->setText("");
                                                 preview_ptr->setPixmap(QPixmap::fromImage(img));
                                                 report_ptr->setPlainText(text);
                                             },Qt::QueuedConnection);
                                     }).detach();
                     });
}

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
    dlg.setAcceptMode(accept);
    dlg.setFileMode(mode);
    dlg.setNameFilter(filter);
    dlg.setSidebarUrls(tipl::qt::working_dirs);
    dlg.setDirectory(dir);
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
auto open_image_files(T* parent,QString path,QString filter,bool multiple = true)
{
    return tipl::qt::details::image_dialog(parent,path,filter,QFileDialog::AcceptOpen,
                                           multiple ? QFileDialog::ExistingFiles : QFileDialog::ExistingFile);
}

template<typename T>
auto open_image_file(T* parent,QString path,QString filter)
{
    auto files = open_image_files(parent,path,filter,false);
    return files.isEmpty() ? QString() : files.front();
}

template<typename T>
auto save_image_file(T* parent,QString default_name,QString filter)
{
    auto files = tipl::qt::details::image_dialog(parent,default_name,filter,QFileDialog::AcceptSave,QFileDialog::AnyFile,
                                                 filter.contains("nii.gz") ? QString("nii.gz") : QString());
    return files.isEmpty() ? QString() : files.front();
}

}


#endif//TIPL_QT_EXT2_HPP

#endif//TIPL_GZ_STREAM_HPP


#endif//QIMAGE_H
