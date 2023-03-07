#ifdef QGRAPHICSSCENE_H

#ifndef TIPL_QT_EXT_HPP
#define TIPL_QT_EXT_HPP

#include <QGraphicsView>
#include <QScrollBar>


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

}//qt
}//tipl

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
        scene.addPixmap(tipl::qt::image2pixelmap(I));
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



#endif//TIPL_QT_EXT_HPP
#endif//QGRAPHICSSCENE_H
