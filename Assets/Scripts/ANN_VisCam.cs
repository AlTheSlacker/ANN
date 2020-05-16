using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN_VisCam : MonoBehaviour
{

    [SerializeField] private float speed = 50.0f;
    [SerializeField] private float zoomSpeed = 100.0f;

    void Update()
    {
        Vector3 transCam = new Vector3(Input.GetAxis("Horizontal"), Input.GetAxis("Vertical"), Input.GetAxis("Mouse ScrollWheel") * zoomSpeed);
        transCam = transCam * Time.deltaTime * speed;
        transform.Translate(transCam);
    }

}
